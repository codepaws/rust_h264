//! x86-64 SIMD kernels for motion compensation, mirroring the aarch64 NEON
//! approach in `inter_pred.rs`.
//!
//! All kernels are bit-exact with the scalar reference implementations:
//! - the 6-tap half-pel FIR accumulates in i16 (max |accumulator| is
//!   42·255 = 10,710, +16 rounding still fits),
//! - the diagonal filter stages its horizontal pass in i16 (±10,710) and
//!   finishes the vertical pass in i32 lanes (±449,820),
//! - quarter-pel averaging saturates each half-pel result to u8 *before*
//!   `pavgb` ((a+b+1)>>1), matching the scalar clip-then-avg order,
//! - chroma bilinear weights sum to 64, so `(sum + 32) >> 6` is always in
//!   0..=255 and needs no saturation clamp.
//!
//! Loads are sized so SIMD chunks never read past the slice lengths callers
//! guarantee (`w + 5` bytes for horizontal-filter sources, `w` for vertical
//! row sets, `w + 1` per chroma row); tails below the chunk width run the
//! scalar reference. Luma blocks are at most 16 wide, so kernels process
//! 16-wide chunks, then one 8-wide chunk, then the scalar tail. AVX2 is used
//! for the 32-wide byte averaging in bi-prediction; the FIR kernels gain
//! nothing from 32-wide at these block sizes.

use std::arch::x86_64::*;

/// Detected CPU capability level, in ascending order. Dispatch happens once
/// per kernel call against this cached value — no per-macroblock detection.
/// SSE2 needs no runtime check (it is part of the x86-64 baseline); kernels
/// using newer instructions (`pmaddubsw` in chroma, wider-than-128-bit
/// `pavgb` in bi-pred) gate on the higher levels and fall back otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SimdLevel {
    Scalar,
    Sse2,
    Ssse3,
    Sse41,
    Avx2,
    Avx512,
}

use std::sync::atomic::{AtomicBool, Ordering};

static FORCE_SCALAR: AtomicBool = AtomicBool::new(false);
static LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();

/// Force (or re-enable) scalar code paths process-wide.
pub fn set_force_scalar(force: bool) {
    FORCE_SCALAR.store(force, Ordering::Release);
}

/// True when scalar paths have been forced via [`set_force_scalar`].
pub fn is_forced_scalar() -> bool {
    FORCE_SCALAR.load(Ordering::Acquire)
}

fn detect_level() -> SimdLevel {
    // AVX-512 needs BW (byte/word ops) + VL (128/256-bit forms) + F to be
    // useful for integer pixel work.
    if std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512f")
    {
        SimdLevel::Avx512
    } else if std::arch::is_x86_feature_detected!("avx2") {
        SimdLevel::Avx2
    } else if std::arch::is_x86_feature_detected!("sse4.1") {
        SimdLevel::Sse41
    } else if std::arch::is_x86_feature_detected!("ssse3") {
        SimdLevel::Ssse3
    } else {
        // SSE2 is part of the x86-64 baseline.
        SimdLevel::Sse2
    }
}

/// Effective SIMD level: detected level, or Scalar when forced.
pub(crate) fn level() -> SimdLevel {
    if is_forced_scalar() {
        return SimdLevel::Scalar;
    }
    *LEVEL.get_or_init(detect_level)
}

static AVX2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static AVX2_INIT: std::sync::Once = std::sync::Once::new();

/// Cheap AVX2 availability for per-chunk kernel selection inside unchecked
/// `_simd` bodies: a one-time detection cached in a relaxed atomic load
/// (checked per 16-pixel chunk, not per row).
#[inline(always)]
fn avx2() -> bool {
    AVX2_INIT.call_once(|| {
        let on = std::arch::is_x86_feature_detected!("avx2");
        AVX2.store(on, Ordering::Release);
    });
    AVX2.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Shared building blocks (all bit-exact against the scalar references)
// ---------------------------------------------------------------------------

/// Widen the low 8 bytes of `x` to i16 lanes.
#[inline(always)]
fn widen8(x: __m128i) -> __m128i {
    unsafe { _mm_unpacklo_epi8(x, _mm_setzero_si128()) }
}

/// Widen the high 8 bytes of `x` to i16 lanes.
#[inline(always)]
fn widen8_hi(x: __m128i) -> __m128i {
    unsafe { _mm_unpackhi_epi8(x, _mm_setzero_si128()) }
}

/// 6-tap FIR accumulate over six already-widened i16x8 vectors:
/// `s0 - 5·s1 + 20·s2 + 20·s3 - 5·s4 + s5`, in i16 without rounding.
#[inline(always)]
unsafe fn fir6_acc_w(w: [__m128i; 6]) -> __m128i {
    let t02 = _mm_add_epi16(w[0], w[5]);
    let t13 = _mm_add_epi16(w[1], w[4]);
    let t23 = _mm_add_epi16(w[2], w[3]);
    let c20 = _mm_set1_epi16(20);
    let c5 = _mm_set1_epi16(5);
    _mm_add_epi16(
        t02,
        _mm_sub_epi16(_mm_mullo_epi16(t23, c20), _mm_mullo_epi16(t13, c5)),
    )
}

/// Round `(acc + 16) >> 5` and saturate an i16x8 accumulator to u8x8
/// (result in the low 8 bytes).
#[inline(always)]
unsafe fn round_pack5(acc: __m128i) -> __m128i {
    let r = _mm_srai_epi16(_mm_add_epi16(acc, _mm_set1_epi16(16)), 5);
    _mm_packus_epi16(r, r)
}

/// 6-tap horizontal FIR of one 8-wide chunk from `src[i..]`.
/// Caller guarantees `i + 8 + 5 <= src.len()`.
#[inline(always)]
unsafe fn fir6_h8(src: &[u8], i: usize) -> __m128i {
    let p = src.as_ptr().add(i);
    fir6_acc_w([
        widen8(_mm_loadl_epi64(p as *const __m128i)),
        widen8(_mm_loadl_epi64(p.add(1) as *const __m128i)),
        widen8(_mm_loadl_epi64(p.add(2) as *const __m128i)),
        widen8(_mm_loadl_epi64(p.add(3) as *const __m128i)),
        widen8(_mm_loadl_epi64(p.add(4) as *const __m128i)),
        widen8(_mm_loadl_epi64(p.add(5) as *const __m128i)),
    ])
}

/// 6-tap vertical FIR of one 8-wide chunk from six row slices at offset `i`.
/// Caller guarantees `i + 8 <= rows[k].len()`.
#[inline(always)]
unsafe fn fir6_v8(rows: [&[u8]; 6], i: usize) -> __m128i {
    fir6_acc_w([
        widen8(_mm_loadl_epi64(rows[0].as_ptr().add(i) as *const __m128i)),
        widen8(_mm_loadl_epi64(rows[1].as_ptr().add(i) as *const __m128i)),
        widen8(_mm_loadl_epi64(rows[2].as_ptr().add(i) as *const __m128i)),
        widen8(_mm_loadl_epi64(rows[3].as_ptr().add(i) as *const __m128i)),
        widen8(_mm_loadl_epi64(rows[4].as_ptr().add(i) as *const __m128i)),
        widen8(_mm_loadl_epi64(rows[5].as_ptr().add(i) as *const __m128i)),
    ])
}

/// Store the low 8 bytes of `x` to `out[i..i+8]`.
#[inline(always)]
unsafe fn store8(out: &mut [u8], i: usize, x: __m128i) {
    _mm_storel_epi64(out.as_mut_ptr().add(i) as *mut __m128i, x);
}

/// `pavgb` of the low 8 bytes of two vectors.
#[inline(always)]
unsafe fn avg8(a: __m128i, b: __m128i) -> __m128i {
    _mm_avg_epu8(a, b)
}

/// Stage-1 horizontal FIR (unrounded i16) of six rows at chunk offset `i`.
/// Each row must have `i + 8 + 5` accessible bytes.
#[inline(always)]
unsafe fn hv_stage1(rows_hv: &[&[u8]; 6], i: usize) -> [[i16; 8]; 6] {
    let mut h = [[0i16; 8]; 6];
    for (k, row) in rows_hv.iter().enumerate() {
        let acc = fir6_h8(row, i);
        _mm_storeu_si128(h[k].as_mut_ptr() as *mut __m128i, acc);
    }
    h
}

/// Vertical FIR over six staged i16x8 rows (i16 lanes from stage-1),
/// rounds `(v + 512) >> 10` and clips to u8. Result in the low 8 bytes.
///
/// madd turns 8 i16 lanes into 4 i32 lanes, so the low and high halves of
/// each row pair are accumulated separately (sum_lo = outputs 0..3,
/// sum_hi = outputs 4..7) and concatenated by the final packs_epi32.
#[inline(always)]
unsafe fn hv_vertical(h: &[[i16; 8]; 6]) -> __m128i {
    let load = |k: usize| _mm_loadu_si128(h[k].as_ptr() as *const __m128i);
    // madd weights as i16 pairs packed into i32 lanes. The [-5, 1] pair
    // must be assembled with masked halves: `-5i32 | (1 << 16)` is a no-op
    // on the already-set high bits of the i32 -5.
    let w01 = _mm_set1_epi32(1i32 | ((-5i32) << 16)); // [1, -5]
    let w23 = _mm_set1_epi32(20 | (20i32 << 16)); // [20, 20]
    let w45 = _mm_set1_epi32((-5i32 & 0xFFFF) | (1i32 << 16)); // [-5, 1]
    let mut sum_lo = _mm_madd_epi16(_mm_unpacklo_epi16(load(0), load(1)), w01);
    let mut sum_hi = _mm_madd_epi16(_mm_unpackhi_epi16(load(0), load(1)), w01);
    sum_lo = _mm_add_epi32(sum_lo, _mm_madd_epi16(_mm_unpacklo_epi16(load(2), load(3)), w23));
    sum_hi = _mm_add_epi32(sum_hi, _mm_madd_epi16(_mm_unpackhi_epi16(load(2), load(3)), w23));
    sum_lo = _mm_add_epi32(sum_lo, _mm_madd_epi16(_mm_unpacklo_epi16(load(4), load(5)), w45));
    sum_hi = _mm_add_epi32(sum_hi, _mm_madd_epi16(_mm_unpackhi_epi16(load(4), load(5)), w45));
    let round = |s: __m128i| _mm_srai_epi32(_mm_add_epi32(s, _mm_set1_epi32(512)), 10);
    // packs_epi32 concatenates the two 4-lane halves into i16 lanes
    // [o0..o7]; packus saturates to u8.
    _mm_packus_epi16(
        _mm_packs_epi32(round(sum_lo), round(sum_hi)),
        _mm_setzero_si128(),
    )
}

// ---------------------------------------------------------------------------
// Scalar references (shared by dispatch fallbacks and differential tests)
// ---------------------------------------------------------------------------

#[inline(always)]
fn fir6_scalar(s: &[u8], i: usize) -> i32 {
    s[i] as i32 - 5 * s[i + 1] as i32 + 20 * s[i + 2] as i32 + 20 * s[i + 3] as i32
        - 5 * s[i + 4] as i32
        + s[i + 5] as i32
}

#[inline(always)]
fn fir6_v_scalar(rows: [&[u8]; 6], i: usize) -> i32 {
    rows[0][i] as i32 - 5 * rows[1][i] as i32 + 20 * rows[2][i] as i32
        + 20 * rows[3][i] as i32
        - 5 * rows[4][i] as i32
        + rows[5][i] as i32
}

#[inline(always)]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline(always)]
fn avg_scalar(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

fn scalar_row_half_pel_h(src: &[u8], out: &mut [u8], w: usize) {
    for i in 0..w {
        out[i] = clip_u8((fir6_scalar(src, i) + 16) >> 5);
    }
}

fn scalar_row_half_pel_v(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    for i in 0..w {
        out[i] = clip_u8((fir6_v_scalar(rows, i) + 16) >> 5);
    }
}

fn scalar_row_half_pel_hv(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    for i in 0..w {
        let mut h = [0i32; 6];
        for (k, row) in rows.iter().enumerate() {
            h[k] = fir6_scalar(row, i);
        }
        let v = h[0] - 5 * h[1] + 20 * h[2] + 20 * h[3] - 5 * h[4] + h[5];
        out[i] = clip_u8((v + 512) >> 10);
    }
}

// ---------------------------------------------------------------------------
// Public kernels (crate-internal). Each dispatches on `level()`.
// ---------------------------------------------------------------------------

/// Horizontal half-pel FIR for a row of `w` pixels; `src` has `w + 5` bytes.
#[inline(never)]
pub(crate) fn row_half_pel_h(src: &[u8], out: &mut [u8], w: usize) {
    if level() == SimdLevel::Scalar {
        scalar_row_half_pel_h(src, out, w);
        return;
    }
    unsafe { row_half_pel_h_simd(src, out, w) };
}

/// Unchecked SIMD body of [`row_half_pel_h`] — callers must have resolved
/// the level once (frame/block-dispatched) and hold the SSE2 baseline.
#[inline(always)]
pub(crate) unsafe fn row_half_pel_h_simd(src: &[u8], out: &mut [u8], w: usize) {
    let mut i = 0;
    unsafe {
        while i + 16 <= w {
            if avx2() {
                let p = src.as_ptr().add(i);
                half_pel_h16_avx2(p, out.as_mut_ptr().add(i));
            } else {
                let p = src.as_ptr().add(i);
                let l = [
                    _mm_loadu_si128(p as *const __m128i),
                    _mm_loadu_si128(p.add(1) as *const __m128i),
                    _mm_loadu_si128(p.add(2) as *const __m128i),
                    _mm_loadu_si128(p.add(3) as *const __m128i),
                    _mm_loadu_si128(p.add(4) as *const __m128i),
                    _mm_loadu_si128(p.add(5) as *const __m128i),
                ];
                let lo = fir6_acc_w(l.map(widen8));
                let hi = fir6_acc_w(l.map(widen8_hi));
                let packed = _mm_packus_epi16(
                    _mm_srai_epi16(_mm_add_epi16(lo, _mm_set1_epi16(16)), 5),
                    _mm_srai_epi16(_mm_add_epi16(hi, _mm_set1_epi16(16)), 5),
                );
                _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, packed);
            }
            i += 16;
        }
        if i + 8 <= w {
            let acc = fir6_h8(src, i);
            store8(out, i, round_pack5(acc));
            i += 8;
        }
    }
    while i < w {
        out[i] = clip_u8((fir6_scalar(src, i) + 16) >> 5);
        i += 1;
    }
}

/// AVX2 horizontal half-pel FIR for one 16-wide chunk: the six 16-byte
/// loads widen to full 16-lane i16 vectors in one `vpmovzxbw` each, so the
/// whole FIR runs in single YMM registers instead of two XMM halves.
/// Caller guarantees `p..p+21` and `out..out+16` valid.
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn half_pel_h16_avx2(p: *const u8, out: *mut u8) {
    unsafe {
        let load = |k: usize| _mm256_cvtepu8_epi16(_mm_loadu_si128(p.add(k) as *const __m128i));
        let s0 = load(0);
        let s1 = load(1);
        let s2 = load(2);
        let s3 = load(3);
        let s4 = load(4);
        let s5 = load(5);
        // acc = (s0 + s5) + 20*(s2 + s3) - 5*(s1 + s4), i16-safe (max 10,710).
        let t02 = _mm256_add_epi16(s0, s5);
        let t13 = _mm256_add_epi16(s1, s4);
        let t23 = _mm256_add_epi16(s2, s3);
        let c20 = _mm256_set1_epi16(20);
        let c5 = _mm256_set1_epi16(5);
        let acc = _mm256_add_epi16(
            t02,
            _mm256_sub_epi16(_mm256_mullo_epi16(t23, c20), _mm256_mullo_epi16(t13, c5)),
        );
        let r = _mm256_srai_epi16(_mm256_add_epi16(acc, _mm256_set1_epi16(16)), 5);
        // packus(a, a) lays out qwords [L, L, H, H] (L/H = saturated low/high
        // 8 lanes); the contiguous 16-byte result is q0 + q2 → imm 0x28.
        let packed = _mm256_packus_epi16(r, r);
        let fixed = _mm256_permute4x64_epi64(packed, 0x28);
        _mm_storeu_si128(out as *mut __m128i, _mm256_castsi256_si128(fixed));
    }
}

/// Vertical half-pel FIR for a row of `w` pixels; each of `rows` has `w` bytes.
#[inline(never)]
pub(crate) fn row_half_pel_v(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    if level() == SimdLevel::Scalar {
        scalar_row_half_pel_v(rows, out, w);
        return;
    }
    unsafe { row_half_pel_v_simd(rows, out, w) };
}

/// Unchecked SIMD body of [`row_half_pel_v`] (see `row_half_pel_h_simd`).
#[inline(always)]
pub(crate) unsafe fn row_half_pel_v_simd(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    let mut i = 0;
    unsafe {
        while i + 16 <= w {
            if avx2() {
                half_pel_v16_avx2(&rows, i, out.as_mut_ptr().add(i));
            } else {
                let l: [__m128i; 6] = rows.map(|r| {
                    _mm_loadu_si128(r.as_ptr().add(i) as *const __m128i)
                });
                let lo = fir6_acc_w(l.map(widen8));
                let hi = fir6_acc_w(l.map(widen8_hi));
                let packed = _mm_packus_epi16(
                    _mm_srai_epi16(_mm_add_epi16(lo, _mm_set1_epi16(16)), 5),
                    _mm_srai_epi16(_mm_add_epi16(hi, _mm_set1_epi16(16)), 5),
                );
                _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, packed);
            }
            i += 16;
        }
        if i + 8 <= w {
            let acc = fir6_v8(rows, i);
            store8(out, i, round_pack5(acc));
            i += 8;
        }
    }
    while i < w {
        out[i] = clip_u8((fir6_v_scalar(rows, i) + 16) >> 5);
        i += 1;
    }
}

/// AVX2 vertical half-pel FIR for one 16-wide chunk from six row slices at
/// byte offset `i`: same YMM-widened FIR as the horizontal kernel, with the
/// six loads coming from the six rows. Caller guarantees `i + 16 <= row len`.
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn half_pel_v16_avx2(rows: &[&[u8]; 6], i: usize, out: *mut u8) {
    unsafe {
        let load = |r: &[u8]| _mm256_cvtepu8_epi16(_mm_loadu_si128(r.as_ptr().add(i) as *const __m128i));
        let s0 = load(rows[0]);
        let s1 = load(rows[1]);
        let s2 = load(rows[2]);
        let s3 = load(rows[3]);
        let s4 = load(rows[4]);
        let s5 = load(rows[5]);
        let t02 = _mm256_add_epi16(s0, s5);
        let t13 = _mm256_add_epi16(s1, s4);
        let t23 = _mm256_add_epi16(s2, s3);
        let c20 = _mm256_set1_epi16(20);
        let c5 = _mm256_set1_epi16(5);
        let acc = _mm256_add_epi16(
            t02,
            _mm256_sub_epi16(_mm256_mullo_epi16(t23, c20), _mm256_mullo_epi16(t13, c5)),
        );
        let r = _mm256_srai_epi16(_mm256_add_epi16(acc, _mm256_set1_epi16(16)), 5);
        // Same pack layout as the horizontal kernel: [L, L, H, H] -> q0 + q2.
        let packed = _mm256_packus_epi16(r, r);
        let fixed = _mm256_permute4x64_epi64(packed, 0x28);
        _mm_storeu_si128(out as *mut __m128i, _mm256_castsi256_si128(fixed));
    }
}

/// Diagonal half-pel FIR for a row of `w` pixels; each of `rows` has
/// `w + 5` bytes. Two-stage: horizontal into i16, vertical in i32.
#[inline(never)]
pub(crate) fn row_half_pel_hv(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    if level() == SimdLevel::Scalar {
        scalar_row_half_pel_hv(rows, out, w);
        return;
    }
    unsafe { row_half_pel_hv_simd(rows, out, w) };
}

/// Unchecked SIMD body of [`row_half_pel_hv`] (see `row_half_pel_h_simd`).
#[inline(always)]
pub(crate) unsafe fn row_half_pel_hv_simd(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    let mut i = 0;
    unsafe {
        while i + 8 <= w {
            let h = hv_stage1(&rows, i);
            let packed = hv_vertical(&h);
            store8(out, i, packed);
            i += 8;
        }
    }
    while i < w {
        let mut hh = [0i32; 6];
        for (k, row) in rows.iter().enumerate() {
            hh[k] = fir6_scalar(row, i);
        }
        let v = hh[0] - 5 * hh[1] + 20 * hh[2] + 20 * hh[3] - 5 * hh[4] + hh[5];
        out[i] = clip_u8((v + 512) >> 10);
        i += 1;
    }
}

/// Quarter-pel: `avg(integer_row, h_half_pel)` — arms (1,0).
#[inline(never)]
pub(crate) fn row_avg_int_h(int_row: &[u8], src_h: &[u8], out: &mut [u8], w: usize, simd: bool) {
    if !simd {
        for i in 0..w {
            let hp = clip_u8((fir6_scalar(src_h, i) + 16) >> 5);
            out[i] = avg_scalar(int_row[i], hp);
        }
        return;
    }
    let mut i = 0;
    unsafe {
        while i + 8 <= w {
            let hp = round_pack5(fir6_h8(src_h, i));
            let int = _mm_loadl_epi64(int_row.as_ptr().add(i) as *const __m128i);
            store8(out, i, avg8(int, hp));
            i += 8;
        }
    }
    while i < w {
        let hp = clip_u8((fir6_scalar(src_h, i) + 16) >> 5);
        out[i] = avg_scalar(int_row[i], hp);
        i += 1;
    }
}

/// Quarter-pel: `avg(h_half_pel, integer_row)` — arm (3,0) (integer row at +1).
#[inline(never)]
pub(crate) fn row_avg_h_int(src_h: &[u8], int_row: &[u8], out: &mut [u8], w: usize, simd: bool) {
    if !simd {
        for i in 0..w {
            let hp = clip_u8((fir6_scalar(src_h, i) + 16) >> 5);
            out[i] = avg_scalar(hp, int_row[i]);
        }
        return;
    }
    let mut i = 0;
    unsafe {
        while i + 8 <= w {
            let hp = round_pack5(fir6_h8(src_h, i));
            let int = _mm_loadl_epi64(int_row.as_ptr().add(i) as *const __m128i);
            store8(out, i, avg8(hp, int));
            i += 8;
        }
    }
    while i < w {
        let hp = clip_u8((fir6_scalar(src_h, i) + 16) >> 5);
        out[i] = avg_scalar(hp, int_row[i]);
        i += 1;
    }
}

/// Quarter-pel: `avg(integer_row, v_half_pel)` — arms (0,1) and (0,3).
#[inline(never)]
pub(crate) fn row_avg_int_v(
    int_row: &[u8],
    rows: [&[u8]; 6],
    out: &mut [u8],
    w: usize,
    simd: bool,
) {
    if !simd {
        for i in 0..w {
            let hp = clip_u8((fir6_v_scalar(rows, i) + 16) >> 5);
            out[i] = avg_scalar(int_row[i], hp);
        }
        return;
    }
    let mut i = 0;
    unsafe {
        while i + 8 <= w {
            let vp = round_pack5(fir6_v8(rows, i));
            let int = _mm_loadl_epi64(int_row.as_ptr().add(i) as *const __m128i);
            store8(out, i, avg8(int, vp));
            i += 8;
        }
    }
    while i < w {
        let hp = clip_u8((fir6_v_scalar(rows, i) + 16) >> 5);
        out[i] = avg_scalar(int_row[i], hp);
        i += 1;
    }
}

/// Quarter-pel: `avg(h_half_pel, v_half_pel)` — arms (1,1), (3,1), (1,3), (3,3).
#[inline(never)]
pub(crate) fn row_avg_h_v(
    src_h: &[u8],
    rows_v: [&[u8]; 6],
    out: &mut [u8],
    w: usize,
    simd: bool,
) {
    if !simd {
        for i in 0..w {
            let hp = clip_u8((fir6_scalar(src_h, i) + 16) >> 5);
            let vp = clip_u8((fir6_v_scalar(rows_v, i) + 16) >> 5);
            out[i] = avg_scalar(hp, vp);
        }
        return;
    }
    let mut i = 0;
    unsafe {
        while i + 8 <= w {
            let hp = round_pack5(fir6_h8(src_h, i));
            let vp = round_pack5(fir6_v8(rows_v, i));
            store8(out, i, avg8(hp, vp));
            i += 8;
        }
    }
    while i < w {
        let hp = clip_u8((fir6_scalar(src_h, i) + 16) >> 5);
        let vp = clip_u8((fir6_v_scalar(rows_v, i) + 16) >> 5);
        out[i] = avg_scalar(hp, vp);
        i += 1;
    }
}

/// Quarter-pel: `avg(h_half_pel, hv_half_pel)` — arms (2,1) and (2,3).
#[inline(never)]
pub(crate) fn row_avg_h_hv(
    src_h: &[u8],
    rows_hv: [&[u8]; 6],
    out: &mut [u8],
    w: usize,
    simd: bool,
) {
    if !simd {
        for i in 0..w {
            let hp = clip_u8((fir6_scalar(src_h, i) + 16) >> 5);
            let mut hh = [0i32; 6];
            for (k, row) in rows_hv.iter().enumerate() {
                hh[k] = fir6_scalar(row, i);
            }
            let hv = hh[0] - 5 * hh[1] + 20 * hh[2] + 20 * hh[3] - 5 * hh[4] + hh[5];
            out[i] = avg_scalar(hp, clip_u8((hv + 512) >> 10));
        }
        return;
    }
    let mut i = 0;
    unsafe {
        while i + 8 <= w {
            let hp = round_pack5(fir6_h8(src_h, i));
            let hvp = hv_vertical(&hv_stage1(&rows_hv, i));
            store8(out, i, avg8(hp, hvp));
            i += 8;
        }
    }
    while i < w {
        let hp = clip_u8((fir6_scalar(src_h, i) + 16) >> 5);
        let mut hh = [0i32; 6];
        for (k, row) in rows_hv.iter().enumerate() {
            hh[k] = fir6_scalar(row, i);
        }
        let hv = hh[0] - 5 * hh[1] + 20 * hh[2] + 20 * hh[3] - 5 * hh[4] + hh[5];
        out[i] = avg_scalar(hp, clip_u8((hv + 512) >> 10));
        i += 1;
    }
}

/// Quarter-pel: `avg(v_half_pel, hv_half_pel)` — arms (1,2) and (3,2).
#[inline(never)]
pub(crate) fn row_avg_v_hv(
    rows_v: [&[u8]; 6],
    rows_hv: [&[u8]; 6],
    out: &mut [u8],
    w: usize,
    simd: bool,
) {
    if !simd {
        for i in 0..w {
            let vp = clip_u8((fir6_v_scalar(rows_v, i) + 16) >> 5);
            let mut hh = [0i32; 6];
            for (k, row) in rows_hv.iter().enumerate() {
                hh[k] = fir6_scalar(row, i);
            }
            let hv = hh[0] - 5 * hh[1] + 20 * hh[2] + 20 * hh[3] - 5 * hh[4] + hh[5];
            out[i] = avg_scalar(vp, clip_u8((hv + 512) >> 10));
        }
        return;
    }
    let mut i = 0;
    unsafe {
        while i + 8 <= w {
            let vp = round_pack5(fir6_v8(rows_v, i));
            let hvp = hv_vertical(&hv_stage1(&rows_hv, i));
            store8(out, i, avg8(vp, hvp));
            i += 8;
        }
    }
    while i < w {
        let vp = clip_u8((fir6_v_scalar(rows_v, i) + 16) >> 5);
        let mut hh = [0i32; 6];
        for (k, row) in rows_hv.iter().enumerate() {
            hh[k] = fir6_scalar(row, i);
        }
        let hv = hh[0] - 5 * hh[1] + 20 * hh[2] + 20 * hh[3] - 5 * hh[4] + hh[5];
        out[i] = avg_scalar(vp, clip_u8((hv + 512) >> 10));
        i += 1;
    }
}

/// Chroma bilinear interpolation block (in-bounds path). Reads
/// `block_w + 1` bytes from each of `block_h + 1` rows starting at
/// `top_off` in `ref_plane` (row stride `ref_width`). Weights sum to 64;
/// `(sum + 32) >> 6` never exceeds 255.
///
/// The SIMD path uses `pmaddubsw` (SSSE3), so it is `#[target_feature]`-
/// gated and dispatched only when SSSE3 was detected at runtime; SSE2-only
/// CPUs take the scalar path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_bilinear_block(
    ref_plane: &[u8],
    top_off: usize,
    ref_width: usize,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
    c00: u8,
    c01: u8,
    c10: u8,
    c11: u8,
) {
    if level() >= SimdLevel::Ssse3 {
        unsafe { ssse3_chroma_bilinear_block(ref_plane, top_off, ref_width, block_w, block_h, output, c00, c01, c10, c11) }
    } else {
        scalar_chroma_bilinear_block(ref_plane, top_off, ref_width, block_w, block_h, output, c00, c01, c10, c11);
    }
}

#[allow(clippy::too_many_arguments)]
fn scalar_chroma_bilinear_block(
    ref_plane: &[u8],
    top_off: usize,
    ref_width: usize,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
    c00: u8,
    c01: u8,
    c10: u8,
    c11: u8,
) {
    for row in 0..block_h {
        let row_top = top_off + row * ref_width;
        let row_bot = row_top + ref_width;
        for i in 0..block_w {
            let val = c00 as i32 * ref_plane[row_top + i] as i32
                + c01 as i32 * ref_plane[row_top + i + 1] as i32
                + c10 as i32 * ref_plane[row_bot + i] as i32
                + c11 as i32 * ref_plane[row_bot + i + 1] as i32;
            output[row * block_w + i] = ((val + 32) >> 6) as u8;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "ssse3")]
#[inline(never)]
unsafe fn ssse3_chroma_bilinear_block(
    ref_plane: &[u8],
    top_off: usize,
    ref_width: usize,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
    c00: u8,
    c01: u8,
    c10: u8,
    c11: u8,
) {
    // Weight vectors as i8 pairs for maddubs: [c00, c01] and [c10, c11].
    let wa = _mm_set1_epi16(((c01 as i16) << 8) | c00 as i16);
    let wb = _mm_set1_epi16(((c11 as i16) << 8) | c10 as i16);
    for row in 0..block_h {
        let top_p = ref_plane.as_ptr().add(top_off + row * ref_width);
        let bot_p = top_p.add(ref_width);
        let out_p = output.as_mut_ptr().add(row * block_w);
        let mut i = 0;
        while i + 8 <= block_w {
            // Interleave (top[i], top[i+1]) pairs for lane k = pixel i+k.
            let t = _mm_loadl_epi64(top_p.add(i) as *const __m128i);
            let t1 = _mm_loadl_epi64(top_p.add(i + 1) as *const __m128i);
            let ta = _mm_unpacklo_epi8(t, t1);
            let b = _mm_loadl_epi64(bot_p.add(i) as *const __m128i);
            let b1 = _mm_loadl_epi64(bot_p.add(i + 1) as *const __m128i);
            let bb = _mm_unpacklo_epi8(b, b1);
            // (c00·t + c01·t1) + (c10·b + c11·b1), max 64·255 = 16320 (i16-safe)
            let acc = _mm_add_epi16(_mm_maddubs_epi16(ta, wa), _mm_maddubs_epi16(bb, wb));
            let res = _mm_srai_epi16(_mm_add_epi16(acc, _mm_set1_epi16(32)), 6);
            store8(output, row * block_w + i, _mm_packus_epi16(res, res));
            i += 8;
        }
        while i < block_w {
            let val = c00 as i32 * *top_p.add(i) as i32
                + c01 as i32 * *top_p.add(i + 1) as i32
                + c10 as i32 * *bot_p.add(i) as i32
                + c11 as i32 * *bot_p.add(i + 1) as i32;
            *out_p.add(i) = ((val + 32) >> 6) as u8;
            i += 1;
        }
    }
}

/// Bi-prediction byte average over whole buffers:
/// `out[k] = (a[k] + b[k] + 1) >> 1` — exactly `pavgb` at any width.
/// Dispatches to the widest available variant (AVX-512 64-wide, AVX2
/// 32-wide, SSE2 16-wide) with a scalar tail.
pub(crate) fn avg_bytes(a: &[u8], b: &[u8], out: &mut [u8]) {
    let n = out.len().min(a.len()).min(b.len());
    let lvl = level();
    if lvl >= SimdLevel::Avx512 {
        unsafe { avx512_avg_bytes(a, b, out, n) }
    } else if lvl >= SimdLevel::Avx2 {
        unsafe { avx2_avg_bytes(a, b, out, n) }
    } else if lvl >= SimdLevel::Sse2 {
        unsafe { sse2_avg_bytes(a, b, out, n) }
    } else {
        for (o, (&x, &y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
            *o = avg_scalar(x, y);
        }
    }
}

#[target_feature(enable = "sse2")]
#[inline(never)]
unsafe fn sse2_avg_bytes(a: &[u8], b: &[u8], out: &mut [u8], n: usize) {
    let mut i = 0;
    while i + 16 <= n {
        let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, _mm_avg_epu8(va, vb));
        i += 16;
    }
    while i < n {
        out[i] = avg_scalar(a[i], b[i]);
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn avx2_avg_bytes(a: &[u8], b: &[u8], out: &mut [u8], n: usize) {
    let mut i = 0;
    while i + 32 <= n {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, _mm256_avg_epu8(va, vb));
        i += 32;
    }
    while i + 16 <= n {
        let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, _mm_avg_epu8(va, vb));
        i += 16;
    }
    while i < n {
        out[i] = avg_scalar(a[i], b[i]);
        i += 1;
    }
}

#[target_feature(enable = "avx512bw,avx512vl,avx512f")]
#[inline(never)]
unsafe fn avx512_avg_bytes(a: &[u8], b: &[u8], out: &mut [u8], n: usize) {
    let mut i = 0;
    while i + 64 <= n {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
        _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut __m512i, _mm512_avg_epu8(va, vb));
        i += 64;
    }
    // Buffers here are at most 16x16 (256 bytes), so fall back to SSE-wide.
    while i + 16 <= n {
        let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, _mm_avg_epu8(va, vb));
        i += 16;
    }
    while i < n {
        out[i] = avg_scalar(a[i], b[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift PRNG for differential tests.
    struct Rng(u64);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x as u32
        }
        fn byte(&mut self) -> u8 {
            (self.next_u32() >> 24) as u8
        }
        fn byte_biased(&mut self) -> u8 {
            // Bias towards extremes to exercise saturation paths.
            match self.next_u32() % 4 {
                0 => 0,
                1 => 255,
                _ => self.byte(),
            }
        }
    }

    /// Serialize tests that toggle the global force-scalar switch.
    static PATH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_both_paths<T>(f: impl Fn() -> T) -> (T, T) {
        let _guard = PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_force_scalar(true);
        let scalar = f();
        set_force_scalar(false);
        let simd = f();
        (scalar, simd)
    }

    fn cmp(name: &str, w: usize, a: &[u8], b: &[u8]) {
        assert_eq!(a.len(), b.len(), "{name} length mismatch at w={w}");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "{name} mismatch at w={w} index {i}: {x} != {y}");
        }
    }

    #[test]
    fn differential_row_half_pel_h() {
        let mut rng = Rng(0x5eed_1234);
        for w in 1..=33usize {
            let src: Vec<u8> = (0..w + 5).map(|_| rng.byte_biased()).collect();
            let (a, b) = with_both_paths(|| {
                let mut out = vec![0u8; w];
                row_half_pel_h(&src, &mut out, w);
                out
            });
            cmp("row_half_pel_h", w, &a, &b);
        }
    }

    #[test]
    fn differential_row_half_pel_v() {
        let mut rng = Rng(0x5eed_5678);
        for w in 1..=33usize {
            let rows: [&[u8]; 6] = [
                &(0..w).map(|_| rng.byte_biased()).collect::<Vec<_>>(),
                &(0..w).map(|_| rng.byte_biased()).collect::<Vec<_>>(),
                &(0..w).map(|_| rng.byte_biased()).collect::<Vec<_>>(),
                &(0..w).map(|_| rng.byte_biased()).collect::<Vec<_>>(),
                &(0..w).map(|_| rng.byte_biased()).collect::<Vec<_>>(),
                &(0..w).map(|_| rng.byte_biased()).collect::<Vec<_>>(),
            ];
            let (a, b) = with_both_paths(|| {
                let mut out = vec![0u8; w];
                row_half_pel_v(rows, &mut out, w);
                out
            });
            cmp("row_half_pel_v", w, &a, &b);
        }
    }

    #[test]
    fn differential_row_half_pel_hv() {
        let mut rng = Rng(0x5eed_9abc);
        for w in 1..=33usize {
            let bufs: Vec<Vec<u8>> =
                (0..6).map(|_| (0..w + 5).map(|_| rng.byte_biased()).collect()).collect();
            let rows: [&[u8]; 6] = [
                &bufs[0], &bufs[1], &bufs[2], &bufs[3], &bufs[4], &bufs[5],
            ];
            let (a, b) = with_both_paths(|| {
                let mut out = vec![0u8; w];
                row_half_pel_hv(rows, &mut out, w);
                out
            });
            cmp("row_half_pel_hv", w, &a, &b);
        }
    }

    #[test]
    fn differential_qpel_avg_kernels() {
        let mut rng = Rng(0x5eed_def0);
        for w in 1..=33usize {
            let int_row: Vec<u8> = (0..w).map(|_| rng.byte_biased()).collect();
            let src_h: Vec<u8> = (0..w + 5).map(|_| rng.byte_biased()).collect();
            let mut mk_rows = |extra: usize| -> Vec<Vec<u8>> {
                (0..6).map(|_| (0..w + extra).map(|_| rng.byte_biased()).collect()).collect()
            };
            let rv = mk_rows(0);
            let rhv = mk_rows(5);
            let rows_v: [&[u8]; 6] = [&rv[0], &rv[1], &rv[2], &rv[3], &rv[4], &rv[5]];
            let rows_hv: [&[u8]; 6] = [&rhv[0], &rhv[1], &rhv[2], &rhv[3], &rhv[4], &rhv[5]];

            let (a, b) = with_both_paths(|| {
                let mut out = vec![0u8; w];
                row_avg_int_h(&int_row, &src_h, &mut out, w, !is_forced_scalar());
                out
            });
            cmp("row_avg_int_h", w, &a, &b);

            let (a, b) = with_both_paths(|| {
                let mut out = vec![0u8; w];
                row_avg_h_int(&src_h, &int_row, &mut out, w, !is_forced_scalar());
                out
            });
            cmp("row_avg_h_int", w, &a, &b);

            let (a, b) = with_both_paths(|| {
                let mut out = vec![0u8; w];
                row_avg_int_v(&int_row, rows_v, &mut out, w, !is_forced_scalar());
                out
            });
            cmp("row_avg_int_v", w, &a, &b);

            let (a, b) = with_both_paths(|| {
                let mut out = vec![0u8; w];
                row_avg_h_v(&src_h, rows_v, &mut out, w, !is_forced_scalar());
                out
            });
            cmp("row_avg_h_v", w, &a, &b);

            let (a, b) = with_both_paths(|| {
                let mut out = vec![0u8; w];
                row_avg_h_hv(&src_h, rows_hv, &mut out, w, !is_forced_scalar());
                out
            });
            cmp("row_avg_h_hv", w, &a, &b);

            let (a, b) = with_both_paths(|| {
                let mut out = vec![0u8; w];
                row_avg_v_hv(rows_v, rows_hv, &mut out, w, !is_forced_scalar());
                out
            });
            cmp("row_avg_v_hv", w, &a, &b);
        }
    }

    #[test]
    fn differential_chroma_bilinear_block() {
        let mut rng = Rng(0x5eed_1111);
        for frac_x in 0..8i32 {
            for frac_y in 0..8i32 {
                if frac_x == 0 && frac_y == 0 {
                    continue;
                }
                let c00 = ((8 - frac_x) * (8 - frac_y)) as u8;
                let c01 = (frac_x * (8 - frac_y)) as u8;
                let c10 = ((8 - frac_x) * frac_y) as u8;
                let c11 = (frac_x * frac_y) as u8;
                let bw = 8usize;
                let bh = 8usize;
                let w = 64usize;
                let plane: Vec<u8> = (0..(bh + 2) * w).map(|_| rng.byte()).collect();
                let top_off = w + 3; // not row-aligned on purpose
                let (a, b) = with_both_paths(|| {
                    let mut out = vec![0u8; bw * bh];
                    chroma_bilinear_block(&plane, top_off, w, bw, bh, &mut out, c00, c01, c10, c11);
                    out
                });
                cmp("chroma_bilinear_block", frac_x as usize * 8 + frac_y as usize, &a, &b);
            }
        }
    }

    #[test]
    fn differential_avg_bytes() {
        let mut rng = Rng(0x5eed_2222);
        for n in [1usize, 2, 7, 15, 16, 17, 31, 32, 33, 255, 256] {
            let a: Vec<u8> = (0..n).map(|_| rng.byte_biased()).collect();
            let b: Vec<u8> = (0..n).map(|_| rng.byte_biased()).collect();
            let (x, y) = with_both_paths(|| {
                let mut out = vec![0u8; n];
                avg_bytes(&a, &b, &mut out);
                out
            });
            cmp("avg_bytes", n, &x, &y);
        }
    }

    #[test]
    fn saturation_extremes() {
        // All-zero and all-255 inputs across the half-pel kernels.
        for w in [4usize, 8, 16] {
            let zeros_h = vec![0u8; w + 5];
            let maxs_h = vec![255u8; w + 5];
            let zeros_r = vec![0u8; w];
            let maxs_r = vec![255u8; w];
            for src in [&zeros_h, &maxs_h] {
                let (a, b) = with_both_paths(|| {
                    let mut out = vec![0u8; w];
                    row_half_pel_h(src, &mut out, w);
                    out
                });
                cmp("h extreme", w, &a, &b);
            }
            fn mk6(row: &[u8]) -> [&[u8]; 6] {
                [row, row, row, row, row, row]
            }
            for rows in [mk6(&zeros_r), mk6(&maxs_r)] {
                let (a, b) = with_both_paths(|| {
                    let mut out = vec![0u8; w];
                    row_half_pel_v(rows, &mut out, w);
                    out
                });
                cmp("v extreme", w, &a, &b);
            }
            for rows in [mk6(&zeros_h), mk6(&maxs_h)] {
                let (a, b) = with_both_paths(|| {
                    let mut out = vec![0u8; w];
                    row_half_pel_hv(rows, &mut out, w);
                    out
                });
                cmp("hv extreme", w, &a, &b);
            }
        }
    }
}

#[cfg(test)]
mod microbench {
    use super::*;
    #[test]
    #[ignore] // diagnostic timing scaffold; run explicitly with --ignored
    fn bench_kernels() {
        let w = 16usize;
        let reps = 2_000_000usize;
        let src: Vec<u8> = (0..w + 5).map(|i| (i * 37) as u8).collect();
        let rows: Vec<Vec<u8>> = (0..6).map(|k| (0..w + 5).map(|j| (k * 31 + j * 7) as u8).collect()).collect();
        let r6: [&[u8]; 6] = [&rows[0], &rows[1], &rows[2], &rows[3], &rows[4], &rows[5]];
        let mut out = vec![0u8; w];

        let t = std::time::Instant::now();
        set_force_scalar(true);
        for _ in 0..reps { scalar_row_half_pel_h(&src, &mut out, w); }
        let h_scalar = t.elapsed();

        let t = std::time::Instant::now();
        set_force_scalar(false);
        for _ in 0..reps { row_half_pel_h(&src, &mut out, w); }
        let h_simd = t.elapsed();

        let t = std::time::Instant::now();
        set_force_scalar(true);
        for _ in 0..reps { scalar_row_half_pel_v(r6, &mut out, w); }
        let v_scalar = t.elapsed();

        let t = std::time::Instant::now();
        set_force_scalar(false);
        for _ in 0..reps { row_half_pel_v(r6, &mut out, w); }
        let v_simd = t.elapsed();

        let t = std::time::Instant::now();
        set_force_scalar(true);
        for _ in 0..reps { scalar_row_half_pel_hv(r6, &mut out, w); }
        let hv_scalar = t.elapsed();

        let t = std::time::Instant::now();
        set_force_scalar(false);
        for _ in 0..reps { row_half_pel_hv(r6, &mut out, w); }
        let hv_simd = t.elapsed();

        eprintln!("h : scalar {:?}  simd {:?}  ({:.2}x)", h_scalar, h_simd, h_scalar.as_secs_f64() / h_simd.as_secs_f64());
        eprintln!("v : scalar {:?}  simd {:?}  ({:.2}x)", v_scalar, v_simd, v_scalar.as_secs_f64() / v_simd.as_secs_f64());
        eprintln!("hv: scalar {:?}  simd {:?}  ({:.2}x)", hv_scalar, hv_simd, hv_scalar.as_secs_f64() / hv_simd.as_secs_f64());
        set_force_scalar(false);
    }
}
