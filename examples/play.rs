/// Decode an H.264 bitstream and display frames in a window.
///
/// Usage: cargo run --example play -- <input.h264> [--fps N] [--loop] [--scalar] [--threads N]
///
/// Decoding runs on a dedicated thread (bounded queue for backpressure);
/// the main thread paces display using SPS VUI timing when available.
///
/// Keys:
///   Space  pause / resume
///   Right  step one frame (while paused)
///   D      dump the current frame to frame_NNNN.yuv
///   Esc    quit
use minifb::{Key, Window, WindowOptions};
use rust_h264::decoder::{Frame, OrderedDecoder};
use rust_h264::nal::parse_annex_b;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Common surface of OrderedDecoder and ThreadedDecoder for the player.
trait DecodeSink {
    fn decode_nal(&mut self, nal: &rust_h264::nal::NalUnit)
        -> Result<Vec<Frame>, rust_h264::error::DecodeError>;
    fn flush(&mut self) -> Vec<Frame>;
}

impl DecodeSink for OrderedDecoder {
    fn decode_nal(&mut self, nal: &rust_h264::nal::NalUnit)
        -> Result<Vec<Frame>, rust_h264::error::DecodeError> {
        OrderedDecoder::decode_nal(self, nal)
    }
    fn flush(&mut self) -> Vec<Frame> {
        OrderedDecoder::flush(self)
    }
}

impl DecodeSink for rust_h264::threading::ThreadedDecoder {
    fn decode_nal(&mut self, nal: &rust_h264::nal::NalUnit)
        -> Result<Vec<Frame>, rust_h264::error::DecodeError> {
        rust_h264::threading::ThreadedDecoder::decode_nal(self, nal)
    }
    fn flush(&mut self) -> Vec<Frame> {
        rust_h264::threading::ThreadedDecoder::flush(self)
    }
}

/// Statistics shared between the decode thread and the display loop.
#[derive(Default)]
struct Stats {
    frames_decoded: AtomicU64,
    /// Monotonic decode rate measured over the last second (frames).
    decode_fps: AtomicU64,
    /// Frames currently sitting in the bounded queue.
    queued: AtomicU64,
    stopped: AtomicBool,
}

/// Convert YUV420 frame to RGBX pixel buffer for display (buffer reused).
/// Uses the same fixed-point coefficients as the scalar path, so output is
/// bit-identical; AVX2 processes 8 pixels per iteration when available.
fn yuv_to_argb(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
    argb: &mut Vec<u32>,
) {
    argb.clear();
    argb.resize(width * height, 0);
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2")) {
            unsafe { avx2_yuv_to_argb(y, u, v, width, height, argb) }
            return;
        }
    }
    scalar_yuv_to_argb(y, u, v, width, height, argb)
}

fn scalar_yuv_to_argb(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
    argb: &mut Vec<u32>,
) {
    let cw = width / 2;
    for row in 0..height {
        for col in 0..width {
            let y_val = y[row * width + col] as i32;
            let u_val = u[(row / 2) * cw + col / 2] as i32 - 128;
            let v_val = v[(row / 2) * cw + col / 2] as i32 - 128;

            let r = (y_val + ((v_val * 359 + 128) >> 8)).clamp(0, 255) as u32;
            let g = (y_val - ((u_val * 88 + v_val * 183 + 128) >> 8)).clamp(0, 255) as u32;
            let b = (y_val + ((u_val * 454 + 128) >> 8)).clamp(0, 255) as u32;

            argb[row * width + col] = (r << 16) | (g << 8) | b;
        }
    }
}

/// 8 pixels per iteration in i32 lanes. Each chroma sample is duplicated to
/// its two luma pixels via `unpacklo/hi_epi32` pairing. u32 pixels store
/// directly into the minifb buffer, so no byte packing is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_yuv_to_argb(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
    argb: &mut Vec<u32>,
) {
    use std::arch::x86_64::*;
    let cw = width / 2;
    let zero = _mm256_setzero_si256();
    let c128 = _mm256_set1_epi32(128);
    let kr = _mm256_set1_epi32(359);
    let kg_u = _mm256_set1_epi32(88);
    let kg_v = _mm256_set1_epi32(183);
    let kb = _mm256_set1_epi32(454);
    let rnd = _mm256_set1_epi32(128);
    let hi = _mm256_set1_epi32(255);

    for row in 0..height {
        let y_row = y.as_ptr().add(row * width);
        let u_row = u.as_ptr().add((row / 2) * cw);
        let v_row = v.as_ptr().add((row / 2) * cw);
        let out_row = argb.as_mut_ptr().add(row * width);
        let mut col = 0;
        while col + 8 <= width {
            let yv = _mm256_cvtepu8_epi32(_mm_loadl_epi64(y_row.add(col) as *const __m128i));
            // Pair-duplicate 4 chroma samples to 8 lanes.
            let dup = |p: *const u8| {
                let c4 = _mm_cvtepu8_epi32(_mm_loadl_epi64(p.add(col / 2) as *const __m128i));
                let lo = _mm_unpacklo_epi32(c4, c4);
                let hi4 = _mm_unpackhi_epi32(c4, c4);
                _mm256_set_m128i(hi4, lo)
            };
            let uv = _mm256_sub_epi32(dup(u_row), c128);
            let vv = _mm256_sub_epi32(dup(v_row), c128);

            let r = _mm256_add_epi32(
                yv,
                _mm256_srai_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(vv, kr), rnd),
                    8,
                ),
            );
            let g = _mm256_sub_epi32(
                yv,
                _mm256_srai_epi32(
                    _mm256_add_epi32(
                        _mm256_add_epi32(_mm256_mullo_epi32(uv, kg_u), _mm256_mullo_epi32(vv, kg_v)),
                        rnd,
                    ),
                    8,
                ),
            );
            let b = _mm256_add_epi32(
                yv,
                _mm256_srai_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(uv, kb), rnd),
                    8,
                ),
            );

            let clamp = |x: __m256i| _mm256_min_epi32(hi, _mm256_max_epi32(zero, x));
            let rgb = _mm256_or_si256(
                _mm256_or_si256(_mm256_slli_epi32(clamp(r), 16), _mm256_slli_epi32(clamp(g), 8)),
                clamp(b),
            );
            _mm256_storeu_si256(out_row.add(col) as *mut __m256i, rgb);
            col += 8;
        }
        while col < width {
            let y_val = *y_row.add(col) as i32;
            let u_val = *u_row.add(col / 2) as i32 - 128;
            let v_val = *v_row.add(col / 2) as i32 - 128;
            let r = (y_val + ((v_val * 359 + 128) >> 8)).clamp(0, 255) as u32;
            let g = (y_val - ((u_val * 88 + v_val * 183 + 128) >> 8)).clamp(0, 255) as u32;
            let b = (y_val + ((u_val * 454 + 128) >> 8)).clamp(0, 255) as u32;
            *out_row.add(col) = (r << 16) | (g << 8) | b;
            col += 1;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: play <input.h264> [--fps N] [--loop] [--scalar]");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let mut fps_override: Option<f64> = None;
    let mut do_loop = false;
    let mut force_scalar = false;
    let mut threads = 1usize;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--fps" => {
                i += 1;
                fps_override = args[i].parse().ok();
            }
            "--loop" => do_loop = true,
            "--scalar" => force_scalar = true,
            "--threads" => {
                i += 1;
                threads = args[i].parse().unwrap_or(1);
            }
            other => eprintln!("Unknown option: {other}"),
        }
        i += 1;
    }

    let h264_data: &'static [u8] = match std::fs::read(input_path) {
        // Leaked on purpose: the decode thread needs 'static NAL slices and
        // this is a one-shot player, so the file lives for the process.
        Ok(data) => Box::leak(data.into_boxed_slice()),
        Err(e) => {
            eprintln!("Error: cannot read '{input_path}': {e}");
            std::process::exit(1);
        }
    };
    let nals = parse_annex_b(h264_data);

    #[cfg(target_arch = "x86_64")]
    if force_scalar {
        rust_h264::set_force_scalar(true);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = force_scalar;

    // Frame rate: user override wins; SPS VUI probed from the first pass.
    let probe_fps = {
        let mut probe = OrderedDecoder::new();
        for nal in &nals {
            if probe.decode_nal(nal).is_err() {
                break;
            }
            if probe.frame_rate().is_some() {
                break;
            }
        }
        probe.frame_rate_f64()
    };
    let (fps, fps_source) = match (fps_override, probe_fps) {
        (Some(rate), _) => (rate, "user override"),
        (None, Some(rate)) => (rate, "SPS VUI timing"),
        (None, None) => (30.0, "default (no VUI timing)"),
    };

    let stats = Arc::new(Stats::default());
    // Bounded queue: decode-ahead is capped, so pause/backpressure works.
    let (tx, rx) = sync_channel::<Frame>(8);

    let decode_stats = Arc::clone(&stats);
    let decode_nals = nals;
    let decode_path = input_path.clone();
    thread::spawn(move || {
        let mut window_title_source = decode_path.clone();
        window_title_source.truncate(40);
        loop {
            let mut report_count = 0u64;
            let mut last_report = Instant::now();
            let mut decoder: Box<dyn DecodeSink> = if threads > 1 {
                Box::new(rust_h264::threading::ThreadedDecoder::new(threads))
            } else {
                Box::new(OrderedDecoder::new())
            };
            for nal in &decode_nals {
                if decode_stats.stopped.load(Ordering::Relaxed) {
                    return;
                }
                match decoder.decode_nal(nal) {
                    Ok(frames) => {
                        for frame in frames {
                            let queued = decode_stats.queued.load(Ordering::Relaxed);
                            decode_stats.queued.store(queued + 1, Ordering::Relaxed);
                            if tx.send(frame).is_err() {
                                return; // display closed
                            }
                            decode_stats.frames_decoded.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(e) => {
                        eprintln!("Decode error (continuing): {e:?}");
                    }
                }
                report_count += 1;
                let now = Instant::now();
                if now.duration_since(last_report) >= Duration::from_secs(1) {
                    let n = decode_stats.frames_decoded.load(Ordering::Relaxed);
                    decode_stats
                        .decode_fps
                        .store(n - report_count.min(n), Ordering::Relaxed);
                    // report_count reused as "last reported total"
                    report_count = n;
                    last_report = now;
                }
            }
            for frame in decoder.flush() {
                if tx.send(frame).is_err() {
                    return;
                }
                decode_stats.frames_decoded.fetch_add(1, Ordering::Relaxed);
            }
            if !do_loop {
                return; // drop tx → display sees disconnected channel
            }
            eprintln!(
                "Looping {} ({} frames decoded)",
                window_title_source,
                decode_stats.frames_decoded.load(Ordering::Relaxed)
            );
        }
    });

    // Wait for the first frame to size the window.
    let first = match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("No frames decoded.");
            std::process::exit(1);
        }
    };
    let width = first.width as usize;
    let height = first.height as usize;

    let scale = if width <= 128 && height <= 128 {
        4
    } else if width <= 320 && height <= 240 {
        2
    } else {
        1
    };

    let mut window = Window::new(
        &format!("rust_h264 — {input_path}"),
        width * scale,
        height * scale,
        WindowOptions {
            resize: true,
            scale_mode: minifb::ScaleMode::AspectRatioStretch,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create window");

    eprintln!(
        "Playing {} ({}x{}) at {fps:.2} fps ({fps_source}) — threaded decode, display-order",
        input_path, width, height
    );

    let frame_duration = Duration::from_secs_f64(1.0 / fps);
    let mut current = first;
    let mut argb: Vec<u32> = Vec::with_capacity(width * height);
    let mut paused = false;
    let mut prev_space = false;
    let mut prev_step = false;
    let mut prev_dump = false;
    let mut frame_count: u64 = 0;
    let mut display_fps_count: u64 = 0;
    let mut display_fps_time = Instant::now();
    let mut display_fps: u64;
    let mut last_frame_time = Instant::now();

    loop {
        if !window.is_open() || window.is_key_down(Key::Escape) {
            break;
        }

        // Key edge detection
        let space = window.is_key_down(Key::Space);
        if space && !prev_space {
            paused = !paused;
        }
        prev_space = space;

        let step = window.is_key_down(Key::Right);
        let do_step = step && !prev_step;
        prev_step = step;

        let dump = window.is_key_down(Key::D);
        if dump && !prev_dump {
            let n = frame_count;
            let path = format!("frame_{n:04}.yuv");
            let ok = std::fs::write(
                &path,
                current
                    .y
                    .iter()
                    .chain(&current.u)
                    .chain(&current.v)
                    .copied()
                    .collect::<Vec<u8>>(),
            );
            match ok {
                Ok(()) => eprintln!("dumped {path} ({}x{})", current.width, current.height),
                Err(e) => eprintln!("dump failed: {e}"),
            }
        }
        prev_dump = dump;

        // Fetch the next frame unless paused (or stepping one frame).
        let mut got_new = false;
        if !paused || do_step {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(f) => {
                    let queued = stats.queued.load(Ordering::Relaxed);
                    stats.queued.store(queued.saturating_sub(1), Ordering::Relaxed);
                    current = f;
                    got_new = true;
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Decoder still working (or between loop passes).
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // End of stream: hold last frame, keep handling keys.
                }
            }
        }

        if got_new {
            frame_count += 1;
            // Pace display, but keep the window responsive while waiting.
            while Instant::now().duration_since(last_frame_time) < frame_duration {
                window.update();
                if !window.is_open() || window.is_key_down(Key::Escape) {
                    stats.stopped.store(true, Ordering::Relaxed);
                    eprintln!("Played {frame_count} frames");
                    return;
                }
                thread::sleep(Duration::from_millis(1));
            }
            last_frame_time = Instant::now();
        }

        yuv_to_argb(&current.y, &current.u, &current.v, width, height, &mut argb);
        window
            .update_with_buffer(&argb, width, height)
            .expect("failed to update window");

        display_fps_count += 1;
        let now = Instant::now();
        if now.duration_since(display_fps_time) >= Duration::from_secs(1) {
            display_fps = display_fps_count;
            display_fps_count = 0;
            display_fps_time = now;
            window.set_title(&format!(
                "rust_h264 — {} ({}x{}) | decode {} fps | display {} fps | queued {} | frames {}{}",
                input_path,
                width,
                height,
                stats.decode_fps.load(Ordering::Relaxed),
                display_fps,
                stats.queued.load(Ordering::Relaxed),
                stats.frames_decoded.load(Ordering::Relaxed),
                if paused { " | PAUSED" } else { "" }
            ));
        }
    }

    stats.stopped.store(true, Ordering::Relaxed);
    eprintln!("Played {frame_count} frames");
}
