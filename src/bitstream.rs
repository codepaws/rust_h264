/// Bitstream reader for parsing H.264 NAL unit RBSP data.
/// Reads bits left-to-right (MSB first).
///
/// Decode readers own a zero-padded buffer. Header readers can borrow RBSP
/// bytes and treat the same 128-byte suffix as virtual zero padding.
pub struct BitstreamReader<'a> {
    data: std::borrow::Cow<'a, [u8]>,
    data_len: usize, // actual RBSP length (excluding padding)
    byte_offset: usize,
    bit_offset: u8, // 0-7, bits already consumed in current byte
}

/// Number of zero bytes appended after RBSP data.
/// Must be large enough that reading past the end during multi-slice
/// end-of-slice detection doesn't cause index-out-of-bounds panics.
/// 128 bytes = 1024 bits covers the worst case for any single MB decode.
const PADDING: usize = 128;

impl BitstreamReader<'static> {
    pub fn new(rbsp: &[u8]) -> Self {
        let data_len = rbsp.len();
        let mut data = Vec::with_capacity(data_len + PADDING);
        data.extend_from_slice(rbsp);
        data.resize(data_len + PADDING, 0);
        Self {
            data: std::borrow::Cow::Owned(data),
            data_len,
            byte_offset: 0,
            bit_offset: 0,
        }
    }
}

impl<'a> BitstreamReader<'a> {
    /// Parse a header without copying its RBSP. The virtual padding matches `new`.
    pub fn new_borrowed(rbsp: &'a [u8]) -> Self {
        Self {
            data: std::borrow::Cow::Borrowed(rbsp),
            data_len: rbsp.len(),
            byte_offset: 0,
            bit_offset: 0,
        }
    }

    #[inline(always)]
    fn padded_len(&self) -> usize {
        self.data_len + PADDING
    }

    #[inline(always)]
    fn padded_byte(&self, offset: usize) -> u8 {
        if offset < self.data_len { self.data[offset] } else { 0 }
    }

    /// Read a single bit, returning 0 or 1.
    /// Reads zero bits through the padded suffix, then returns an error.
    /// Call `bits_remaining` to check the actual RBSP length before bulk reads.
    #[inline(always)]
    pub fn read_bit(&mut self) -> Result<u8, &'static str> {
        let byte = match &self.data {
            std::borrow::Cow::Owned(data) => *data.get(self.byte_offset).ok_or("end of bitstream")?,
            std::borrow::Cow::Borrowed(data) => {
                if self.byte_offset >= self.padded_len() {
                    return Err("end of bitstream");
                }
                data.get(self.byte_offset).copied().unwrap_or(0)
            }
        };
        let bit = (byte >> (7 - self.bit_offset)) & 1;
        self.bit_offset += 1;
        if self.bit_offset == 8 {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }
        Ok(bit)
    }

    /// Read `n` bits (up to 32) as a u32.
    pub fn read_bits(&mut self, n: u8) -> Result<u32, &'static str> {
        let mut val: u32 = 0;
        for _ in 0..n {
            val = (val << 1) | self.read_bit()? as u32;
        }
        Ok(val)
    }

    /// Peek at the next `n` bits without consuming them. Zero-pads if fewer bits remain.
    pub fn peek_bits(&self, n: u8) -> u32 {
        let mut val: u32 = 0;
        let mut byte_offset = self.byte_offset;
        let mut bit_offset = self.bit_offset;
        for _ in 0..n {
            // Padding guarantees in-bounds for reasonable peek sizes
            if byte_offset < self.padded_len() {
                val = (val << 1) | ((self.padded_byte(byte_offset) >> (7 - bit_offset)) & 1) as u32;
            } else {
                val <<= 1;
            }
            bit_offset += 1;
            if bit_offset == 8 {
                bit_offset = 0;
                byte_offset += 1;
            }
        }
        val
    }

    /// Advance position by `n` bits without reading them.
    pub fn skip_bits(&mut self, n: u8) {
        let total = self.bit_offset as usize + n as usize;
        self.byte_offset += total / 8;
        self.bit_offset = (total % 8) as u8;
    }

    /// Read an unsigned Exp-Golomb coded value (ue(v)).
    pub fn read_ue(&mut self) -> Result<u32, &'static str> {
        let mut leading_zeros: u32 = 0;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err("exp-golomb overflow");
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let suffix = self.read_bits(leading_zeros as u8)?;
        Ok((1 << leading_zeros) - 1 + suffix)
    }

    /// Read a truncated Exp-Golomb coded value (te(v)).
    /// When max == 1, reads a single bit: bit=0 → value=1, bit=1 → value=0.
    /// When max > 1, reads a standard ue(v).
    pub fn read_te(&mut self, max: u32) -> Result<u32, &'static str> {
        if max > 1 {
            self.read_ue()
        } else {
            // Spec 9.1: value = 1 - bit
            Ok(1 - self.read_bit()? as u32)
        }
    }

    /// Read a signed Exp-Golomb coded value (se(v)).
    pub fn read_se(&mut self) -> Result<i32, &'static str> {
        let code = self.read_ue()?;
        let val = code.div_ceil(2) as i32;
        if code % 2 == 0 {
            Ok(-val)
        } else {
            Ok(val)
        }
    }

    /// Returns true if there is more RBSP data before the trailing bits.
    /// The RBSP trailing bits are: a stop bit (1) followed by zero bits to byte-align.
    pub fn more_rbsp_data(&self) -> bool {
        if self.byte_offset >= self.data_len {
            return false;
        }

        // Find the position of the last non-zero byte in the actual data
        let mut last_nz = self.data_len;
        while last_nz > self.byte_offset && self.data[last_nz - 1] == 0 {
            last_nz -= 1;
        }
        if last_nz <= self.byte_offset {
            return false;
        }

        // If we're before the last non-zero byte, there's definitely more data
        if self.byte_offset < last_nz - 1 {
            return true;
        }

        // We're in the last non-zero byte. The RBSP stop bit is the lowest set bit.
        // Everything above it (towards MSB) is data. Check if there are unconsumed
        // data bits above the stop bit.
        let byte = self.data[self.byte_offset];
        let stop_bit = byte.trailing_zeros();
        let data_bits_in_byte = 7 - stop_bit as u8;
        self.bit_offset < data_bits_in_byte
    }

    /// Advance to the next byte boundary.
    pub fn align_to_byte(&mut self) {
        if self.bit_offset != 0 {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }
    }

    pub fn position(&self) -> (usize, u8) {
        (self.byte_offset, self.bit_offset)
    }

    /// Peek at the byte at the current byte_offset (for debugging).
    pub fn peek_byte(&self) -> u8 {
        assert!(self.byte_offset < self.padded_len(), "byte peek past padding");
        self.padded_byte(self.byte_offset)
    }

    /// Peek at bytes including zero padding (for debugging). Borrowed readers
    /// allocate only when the requested range crosses into virtual padding.
    pub fn peek_bytes(&self, n: usize) -> std::borrow::Cow<'_, [u8]> {
        let end = self.byte_offset.saturating_add(n).min(self.padded_len());
        if self.byte_offset >= end {
            return std::borrow::Cow::Borrowed(&[]);
        }
        if end <= self.data.len() {
            return std::borrow::Cow::Borrowed(&self.data[self.byte_offset..end]);
        }
        let mut bytes = Vec::with_capacity(end - self.byte_offset);
        if self.byte_offset < self.data_len {
            bytes.extend_from_slice(&self.data[self.byte_offset..self.data_len]);
        }
        bytes.resize(end - self.byte_offset, 0);
        std::borrow::Cow::Owned(bytes)
    }

    /// Get the byte-aligned position for CABAC initialization.
    /// Aligns to the next byte boundary and returns (byte_offset, reference to data).
    pub fn cabac_start(&mut self) -> (usize, &[u8]) {
        // CABAC starts after cabac_alignment_one_bit + byte alignment
        // The slice header is followed by a 1-bit then byte-aligned CABAC data
        if self.bit_offset != 0 {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }
        (self.byte_offset, &self.data[..self.data_len])
    }

    pub fn bits_remaining(&self) -> usize {
        if self.byte_offset >= self.data_len {
            return 0;
        }
        (self.data_len - self.byte_offset) * 8 - self.bit_offset as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bits() {
        let data = [0b1010_0011, 0b1100_0000];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
        assert_eq!(r.read_bits(4).unwrap(), 0b0011);
        assert_eq!(r.read_bits(2).unwrap(), 0b11);
    }

    #[test]
    fn test_read_ue() {
        // ue(0) = 1 -> code 0
        // ue(1) = 010 -> code 1
        // ue(2) = 011 -> code 2
        // ue(3) = 00100 -> code 3
        // Bit pattern: 1 010 011 00100 = 1010 0110 0100 = 0xA64
        let data = [0xA6, 0x40];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 0);
        assert_eq!(r.read_ue().unwrap(), 1);
        assert_eq!(r.read_ue().unwrap(), 2);
        assert_eq!(r.read_ue().unwrap(), 3);
    }

    #[test]
    fn test_read_se() {
        // se mapping: code 0->0, 1->1, 2->-1, 3->2, 4->-2
        // ue values:   0       1       2       3       4
        // bits:        1      010     011    00100   00101
        // Combined: 1 010 011 00100 00101 = 1010 0110 0100 0010 1
        let data = [0xA6, 0x42, 0x80];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.read_se().unwrap(), 0);
        assert_eq!(r.read_se().unwrap(), 1);
        assert_eq!(r.read_se().unwrap(), -1);
        assert_eq!(r.read_se().unwrap(), 2);
        assert_eq!(r.read_se().unwrap(), -2);
    }

    #[test]
    fn test_padding_allows_overread() {
        // After reading all real data, reads return 0 (from padding) rather than panicking
        let data = [0xFF];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.bits_remaining(), 8);
        assert_eq!(r.read_bits(8).unwrap(), 0xFF);
        assert_eq!(r.bits_remaining(), 0);
        // Reading into padding returns 0 bits without panic
        assert_eq!(r.read_bits(8).unwrap(), 0);
    }

    /// Fuzz regression: reading far past the buffer must return Err, not panic.
    #[test]
    fn test_read_past_buffer_returns_err() {
        let data = [0xAB; 4]; // 4 bytes = 32 bits
        let mut r = BitstreamReader::new(&data);
        // Read the actual data + padding (128 bytes = 1024 bits)
        for _ in 0..(4 + 128) * 8 {
            let _ = r.read_bit(); // may be Ok or Err
        }
        // Now we're past the padded buffer — must return Err, not panic
        assert!(r.read_bit().is_err());
    }

    #[test]
    fn borrowed_reader_matches_padded_reader_at_truncated_boundaries() {
        for data in [&[][..], &[0x80][..], &[0x00, 0x55][..], &[0xff, 0x00, 0x80][..]] {
            let mut padded = BitstreamReader::new(data);
            let mut borrowed = BitstreamReader::new_borrowed(data);
            for _ in 0..(data.len() + PADDING) * 8 + 1 {
                assert_eq!(borrowed.bits_remaining(), padded.bits_remaining());
                assert_eq!(borrowed.more_rbsp_data(), padded.more_rbsp_data());
                assert_eq!(borrowed.peek_bits(12), padded.peek_bits(12));
                assert_eq!(borrowed.read_bit(), padded.read_bit());
            }
        }
    }

    #[test]
    fn borrowed_byte_peeks_match_owned_across_virtual_padding() {
        let data = [0x12, 0x34];
        let mut owned = BitstreamReader::new(&data);
        let mut borrowed = BitstreamReader::new_borrowed(&data);
        for offset in [0, 1, 2, 3, data.len() + PADDING - 1, data.len() + PADDING] {
            owned.byte_offset = offset;
            borrowed.byte_offset = offset;
            if offset < data.len() + PADDING {
                assert_eq!(borrowed.peek_byte(), owned.peek_byte(), "offset={offset}");
            } else {
                assert!(std::panic::catch_unwind(|| owned.peek_byte()).is_err());
                assert!(std::panic::catch_unwind(|| borrowed.peek_byte()).is_err());
            }
            for n in [0, 1, 4, usize::MAX] {
                assert_eq!(borrowed.peek_bytes(n), owned.peek_bytes(n), "offset={offset}, n={n}");
            }
        }
    }
}
