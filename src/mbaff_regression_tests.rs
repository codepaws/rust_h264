use crate::decoder::Decoder;
use crate::nal::parse_annex_b;

#[test]
fn cabac_mbaff_second_pair_row_matches_all_analytic_frames() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let bitstream = std::fs::read(
        root.join("testdata/mbaff_cabac_second_pair_row_64x64_test.h264"),
    )
    .unwrap();
    let expected = std::fs::read(
        root.join("testdata/mbaff_cabac_second_pair_row_64x64_test.yuv"),
    )
    .unwrap();

    let mut decoder = Decoder::new();
    let mut frames = Vec::new();
    for nal in parse_annex_b(&bitstream) {
        if let Some(frame) = decoder.decode_nal(&nal).unwrap() {
            frames.push(frame);
        }
    }
    if let Some(frame) = decoder.flush() {
        frames.push(frame);
    }

    assert_eq!(frames.len(), 4);
    let mut actual = Vec::with_capacity(expected.len());
    for (frame_index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.width, frame.height), (64, 64));
        assert_eq!(frame.pic_order_cnt, (frame_index as i32) * 2);
        actual.extend_from_slice(&frame.y);
        actual.extend_from_slice(&frame.u);
        actual.extend_from_slice(&frame.v);
    }
    assert_eq!(actual.len(), expected.len());
    if let Some(index) = actual.iter().zip(&expected).position(|(a, e)| a != e) {
        let bytes_per_frame = 64 * 64 + 2 * 32 * 32;
        let frame = index / bytes_per_frame;
        let within_frame = index % bytes_per_frame;
        let (plane, sample) = if within_frame < 64 * 64 {
            ("Y", within_frame)
        } else if within_frame < 64 * 64 + 32 * 32 {
            ("U", within_frame - 64 * 64)
        } else {
            ("V", within_frame - 64 * 64 - 32 * 32)
        };
        let plane_width = if plane == "Y" { 64 } else { 32 };
        panic!(
            "frame {frame} {plane} row {}, column {}: got {}, expected {}",
            sample / plane_width,
            sample % plane_width,
            actual[index],
            expected[index]
        );
    }
}

#[test]
fn cabac_mbaff_b_slice_skip_and_direct_branches_match_reference_yuv() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let bitstream = std::fs::read(root.join("testdata/mbaff_cabac_b_test.h264")).unwrap();
    let expected = std::fs::read(root.join("testdata/mbaff_cabac_b_test.yuv")).unwrap();

    crate::decoder::reset_b_mbaff_skip_direct_counts();
    let mut decoder = Decoder::new();
    let mut frames = Vec::new();
    for nal in parse_annex_b(&bitstream) {
        if let Some(frame) = decoder.decode_nal(&nal).unwrap() {
            frames.push(frame);
        }
    }
    if let Some(frame) = decoder.flush() {
        frames.push(frame);
    }

    frames.sort_by_key(|frame| frame.pic_order_cnt);
    assert_eq!(frames.len(), 8);
    let mut actual = Vec::with_capacity(expected.len());
    for frame in &frames {
        assert_eq!((frame.width, frame.height), (64, 64));
        actual.extend_from_slice(&frame.y);
        actual.extend_from_slice(&frame.u);
        actual.extend_from_slice(&frame.v);
    }
    assert_eq!(actual, expected);

    let (skipped, direct) = crate::decoder::b_mbaff_skip_direct_counts();
    assert_eq!(skipped, 27, "B-slice MBAFF B_Skip branch count");
    assert_eq!(direct, 30, "B-slice MBAFF direct macroblock branch count");
}
