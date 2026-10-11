# rust_h264 next steps

Updated 12 October 2026. The skipped-top/signalled-field-mode CAVLC and CABAC coverage gaps, including the larger-picture CABAC above-neighbor mismatch, are closed. The completed work is recorded in a local commit on `codex/todo-closeout`; preserve the remaining performance and specification follow-ups when resuming.

## Completed: second MBAFF pair row

- [x] Reproduced the 64x64 CABAC MBAFF stream and identified output frame index 1 as `frame_num=1`, POC 2, from the reference NAL at byte offset 653 (NAL type 1, `nal_ref_idc=2`).
- [x] Added a byte-exact test for all four frames. It failed before the fix: the first luma mismatch was frame 1, row 32, column 0 (37 instead of 40); additional mismatches appeared across the second pair row.
- [x] Traced MB address 8. Its above-neighbor lookup selected MB 1, so its CABAC skip context treated it as coded and decoded residual `-3` against the POC 0 reference value 40. The correct same-field above member is MB 0, which is skipped.
- [x] Corrected the top field-coded above-neighbor mapping per H.264 (08/2021) clauses 6.4.10 and Table 6-4 of 6.4.12.2: when the current and above pair are field-coded, select `mbAddrB` (the above pair's top MB) and preserve `yM = yN`.
- [x] Added the 64x64 CABAC stream and analytic YUV fixture plus a reproducible generator; provenance and hashes are documented in [testdata/README.md](testdata/README.md).
- [x] Focused byte-exact regression passes after the correction. Full-suite and all-target/all-feature verification results are recorded below.

The 64x32 CABAC fixture covers skipped-top/signalled-field transitions in the first pair row; the 64x64 regression covers the independent above-neighbor skip-context error in the second pair row.

## Further correctness coverage

- [x] Audit and fix the analogous CAVLC skipped-pair ordering and field-mode inference against clauses 7.3.4 and 7.4.4. Regressions cover skipped-top/signalled-field bottom and fully skipped left-inferred field mode; see `target/spec-audit/cavlc-followup.md`.
- [x] Add targeted CABAC B-slice skip/direct coverage with independent expected pixels: the focused regression checks all eight frames of the existing 64x64 fixture and asserts 27 skipped and 30 direct macroblocks.
- [ ] Add field-coded B-slice coverage and combinations involving slice boundaries, fully skipped pairs, and transitions between frame-coded and field-coded pairs where existing fixtures do not demonstrate them. Existing CABAC/CAVLC B MBAFF fixtures contain no field-coded B macroblocks, so this narrower coverage remains open.
- [ ] Make the fully-skipped CAVLC inference regression fail its pixel comparison when the inferred field mode is deliberately forced to frame mode. Two analytic-pattern experiments are recorded in `target/spec-audit/cavlc-followup.md`: a period-four left region with a static period-two skipped region passed baseline but remained pixel-identical under forced-frame inference, while a 64x64 shifted-field candidate failed baseline decoding and was discarded. The checked-in 64x32 fixture still proves true left inference with a branch counter; wrong-mode pixel sensitivity remains unproven.

## Source provenance and performance follow-up

- [x] Complete a bounded follow-up provenance audit of identified chroma, CABAC, and threading references. It confirms historical source-specific citations in commits `48a2c552` and `19affb9f`, and a high-level threading analogy in `efffaf14`; it did not consult FFmpeg implementation source. No upstream/parent remote is configured, the uncommitted working tree cannot be attributed from commit diffs, and this is not a whole-repository provenance guarantee. The findings and scope are recorded in [CLAUDE.md](CLAUDE.md) and `target/spec-audit/provenance-followup.md` (ignored, temporary detail).
- [x] Preserve the source rule in [CLAUDE.md](CLAUDE.md): do not copy, translate, adapt, or consult FFmpeg implementation source for decoder changes. Use the H.264 specification and independent derivations. Binary tools used to generate or validate test data are a separate practice and must be documented honestly.
- [ ] Complete a controlled performance comparison for all benchmark streams in [BENCHMARK.md](BENCHMARK.md). A paired, reversed-order comparison of both 720p streams now has matching baseline/working fingerprints, but timings varied substantially and the 1080p cases were not completed; no general speedup claim is justified. Raw measurements are in ignored `target/spec-audit/perf-followup.txt`.
- [x] Review and package the accumulated changes into a local commit on `codex/todo-closeout`. The commit is local only; no pull request or publishing was performed. The earlier upstream review found no additional change to import; recheck upstream if substantial time has passed.

## Completed coverage work

- [x] Added `testdata/mbaff_cabac_skipped_top_field_bottom_test.h264` and its expected `.yuv`: four 64x32 frames, with 12 skipped-top/coded-bottom pairs signalling field mode.
- [x] Added a focused decoder test that compares all decoded pixels and requires the field-mode transition counter to be positive.
- [x] Added [the reproducible generator](testdata/generate_mbaff_skipped_top_field_bottom.py) and [fixture documentation](testdata/README.md). Expected pixels are generated directly from a simple pattern: even Y rows 40, odd Y rows 200/210/220/230, and U/V 128. An external decoder independently confirms those bytes. FFmpeg/libx264 executables are used for generation and cross-checking; their implementation source was not consulted.
- [x] Regenerated the fixture twice with identical hashes. Encoder versions and hashes are recorded in the fixture documentation; bitstream identity across other encoder versions is not promised.
- [x] Demonstrated failure on the old corpus when requiring this transition. Demonstrated a passing new fixture and a failing isolated mutation that reconstructs the skipped top with frame layout. The mutated run still recorded all 12 target transitions, so the pixel comparison detects the error.
- [x] Completed independent Sol low review of the fixture, generator, and regression.

## Verification at wrap-up

`cargo test --release --lib`: **214 passed, 0 failed, 1 ignored**. The ignored test is a benchmark. `cargo check --all-targets --all-features` and `git diff --check` passed. The focused CABAC MBAFF B-slice regression checks 27 skipped and 30 direct macroblocks against all eight expected frames. `cargo fmt -- --check` could not run because Windows returned `Access is denied (os error 5)` when launching rustfmt.

Detailed investigation reports and the isolated negative-test copy are under `target/spec-audit`. That directory is ignored and may be removed by `cargo clean`; the durable reproduction instructions and important findings are in this file, `CLAUDE.md`, and `testdata/README.md`.
