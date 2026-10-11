# Test Data

## mbaff_cabac_skipped_top_field_bottom_test.h264 / .yuv

A 64x32, four-frame Main-profile CABAC MBAFF stream. Its source has luma 40
on every even row and luma 200, 210, 220, then 230 on odd rows; chroma is
128. The changing odd field causes P-slice pairs with a skipped top
macroblock and a coded bottom macroblock whose `mb_field_decoding_flag` is
one. The decoder's test-only transition counter reports 12 such pairs.
This exercises the field-mode signalling point described by H.264 (08/2021)
clauses 7.3.4 and 7.4.4. The distinct field levels expose a frame/field row
mapping mistake in the decoded output.

Generate both files from the repo root with
`python -B testdata/generate_mbaff_skipped_top_field_bottom.py`. The script uses
FFmpeg's libx264 **executable encoder** for the H.264 stream. The expected
YUV420 file is generated directly from the analytic source samples: even Y
rows are 40, odd Y rows are 200 + 10 × frame index, and all U/V samples are
128. The script independently decodes the stream with FFmpeg's executable
decoder and requires that its output matches those source bytes exactly.
Neither codec's implementation source was consulted or used as decoder
implementation guidance. The bundled stream
was produced with FFmpeg 8.1 and libx264 core 165 r3222M b35605a, one encoder
thread, fixed QP 22, no B-frames, weighted P prediction, or deblocking. The complete command arguments
are in the script. Stream SHA-256:
`5ef1a117a0845f094aba90eda72940fa0b0175b176a98ffd8d8457a08dfe974f`.
YUV SHA-256:
`008e90f53ea7937e8957bb6cca736978641ced690ec8c17307401b369e783b21`.
The expected YUV is 12,288 bytes (4 frames × 3,072 bytes).

## mbaff_cavlc_skipped_top_field_bottom_test.h264 / .yuv

A 64x32, four-frame Main-profile CAVLC MBAFF stream. The analytic source has
luma 40 on every even row and 200, 210, 220, then 230 on odd rows; U and V are
128. P frames exercise a skipped top followed by a coded bottom with a
signalled field mode. The decoder regression checks this branch with a
test-only counter and compares every output sample against the analytic YUV.

Generate the stream and expected pixels with
`python -B testdata/generate_mbaff_cavlc_skipped_top_field_bottom.py`. The
script uses FFmpeg/libx264 executables for encoding and independent decoding;
the expected YUV is constructed directly from the source pattern and must
match the independent decode byte-for-byte. No implementation source was
consulted. Stream SHA-256:
`111c06e2ce85f8b62eff6295b28e58c00ecf05bd0a05b79151f579ea8eaad510`.
YUV SHA-256:
`008e90f53ea7937e8957bb6cca736978641ced690ec8c17307401b369e783b21`.

## mbaff_cavlc_fully_skipped_field_test.h264 / .yuv

A second 64x32, four-frame Main-profile CAVLC MBAFF stream. Its source has
luma 40 on even rows; odd rows are 200 + 10 × frame index in columns 0–15 and
200 elsewhere; U and V are 128. In P frames the changed left pair signals
field mode, while its unchanged right neighbor is fully skipped and must
inherit the left pair's mode under clause 7.4.4. The picture has one pair row,
so the asserted true inference is specifically same-slice left inheritance;
there is no pair above. The test compares every output sample to the analytic
YUV and uses a test-only counter to prove this path was reached.

Generate it with `python -B testdata/generate_mbaff_cavlc_fully_skipped_field.py`. FFmpeg/libx264
executables encode the stream and independently decode it to confirm the
analytic bytes. Stream SHA-256:
`85213d15c0275d5e338ad69044acea02de7ffad1ab3f09c846a3a7ec4fc88a1c`.
YUV SHA-256:
`808a146745aa465490ff33d38aea5ff17e16db568cb14fdf2ead68dd4ba5c88f`.

## mbaff_cabac_second_pair_row_64x64_test.h264 / .yuv

A four-frame 64x64 Main-profile CABAC MBAFF stream with the same analytic
field pattern as the first CABAC fixture: even luma rows are 40, odd rows are
200, 210, 220, then 230, and U/V are 128. The second pair row exercises a
top field-coded macroblock whose above pair is also field-coded. The decoder
regression checks all samples in all four output frames (POCs 0, 2, 4, 6).

The mismatch was caused by selecting the bottom member of the above pair for
that top field-coded macroblock. H.264 (08/2021) clause 6.4.10 defines
`mbAddrB` as the top macroblock of the above pair. In Table 6-4 of clause
6.4.12.2, with `currMbFrameFlag = 0`, `mbIsTopMbFlag = 1`, and
`mbAddrXFrameFlag = 0`, the `yN < 0` case selects `mbAddrN = mbAddrB` and
`yM = yN`: the top member, in the same field. The corrected lookup applies
that selection while preserving the bottom-member mapping for other cases.
The byte-exact regression failed before the correction and passes afterward.

Generate both files into ignored build output with
`python -B testdata/generate_mbaff_cabac_second_pair_row_64x64.py`. Use
`--out testdata` only when intentionally regenerating the checked-in copies.
The script reuses the analytic pattern and encoding arguments from
`generate_mbaff_skipped_top_field_bottom.py`, setting the height to 64. It
uses FFmpeg 8.1 and libx264 executables to encode the bitstream and to perform
an independent decode check against the analytic YUV bytes; no FFmpeg
implementation source was consulted. The checked-in stream SHA-256 is
`2bf8339d30d591b35caca4b2f33ecba3fcb0a6db0b198f36f80a44ef19f7c953` and the
YUV SHA-256 is
`5deb1186d6cc4325e4b9e9a271a2e93c1f69cdda3ac0cac0b2b05b888ec50cfc`.
The expected YUV is 24,576 bytes (4 frames × 6,144 bytes).

## single_frame.h264

A minimal H.264 Annex B bitstream containing a single 16x16 pixel red frame.

Properties:
- Profile: Constrained Baseline (profile_idc=66)
- Level: 1.0
- Pixel format: YUV 4:2:0, 8-bit
- Entropy coding: CAVLC (no CABAC)
- 1 macroblock (I16x16, DC prediction)
- No B-frames, 1 reference frame

NAL unit structure (each prefixed with `00 00 00 01` start code):
1. **SPS** (NAL type 7) - Sequence Parameter Set
2. **PPS** (NAL type 8) - Picture Parameter Set
3. **SEI** (NAL type 6) - x264 encoder metadata
4. **IDR slice** (NAL type 5) - The single I-frame

Generated with:
```
ffmpeg -f lavfi -i "color=c=red:s=16x16:d=0.04:r=25" -vframes 1 \
  -c:v libx264 -profile:v baseline -level 1 \
  -pix_fmt yuv420p -x264-params "keyint=1:bframes=0:ref=1:no-cabac=1" \
  -f h264 single_frame.h264
```

## single_frame.yuv

Raw YUV420p decode of `single_frame.h264`. 384 bytes total for a 16x16 frame:

| Plane | Size | Offset | Value | Description |
|-------|------|--------|-------|-------------|
| Y     | 16x16 = 256 bytes | 0x000 | 0x51 (81)  | Luma |
| U (Cb)| 8x8 = 64 bytes    | 0x100 | 0x5A (90)  | Chroma blue |
| V (Cr)| 8x8 = 64 bytes    | 0x140 | 0xF0 (240) | Chroma red |

These values correspond to the color red in BT.601 YCbCr space.

Generated with:
```
ffmpeg -i single_frame.h264 -pix_fmt yuv420p -f rawvideo single_frame.yuv
```
