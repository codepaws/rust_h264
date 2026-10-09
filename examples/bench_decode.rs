/// Benchmark decoder throughput on an H.264 Annex B bitstream.
///
/// Usage: cargo run --release --example bench_decode -- <input.h264> [--iterations N] [--scalar]
///
/// Decodes the stream in a warmup pass, then times `--iterations` (default 3)
/// passes and reports the best pass. `--scalar` forces scalar code paths so
/// SIMD gains can be A/B measured on the same machine.
use rust_h264::nal::parse_annex_b;
use std::time::Instant;

fn decode_once(nals: &[rust_h264::nal::NalUnit], threads: usize) -> usize {
    let mut count = 0usize;
    if threads <= 1 {
        let mut decoder = rust_h264::decoder::OrderedDecoder::new();
        for nal in nals {
            if let Ok(frames) = decoder.decode_nal(nal) {
                count += frames.len();
            }
        }
        count += decoder.flush().len();
    } else {
        let mut decoder = rust_h264::threading::ThreadedDecoder::new(threads);
        for nal in nals {
            if let Ok(frames) = decoder.decode_nal(nal) {
                count += frames.len();
            }
        }
        count += decoder.flush().len();
    }
    count
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_decode <input.h264> [--iterations N] [--scalar]");
        std::process::exit(1);
    }
    let mut path: Option<&str> = None;
    let mut iterations = 3usize;
    let mut scalar = false;
    let mut threads = 1usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" => {
                i += 1;
                iterations = args[i].parse().unwrap_or(3);
            }
            "--scalar" => scalar = true,
        "--threads" => {
            i += 1;
            threads = args[i].parse().unwrap_or(1);
        }
            other => path = Some(other),
        }
        i += 1;
    }
    let path = path.unwrap_or_else(|| {
        eprintln!("Usage: bench_decode <input.h264> [--iterations N] [--scalar]");
        std::process::exit(1);
    });

    #[cfg(target_arch = "x86_64")]
    if scalar {
        rust_h264::set_force_scalar(true);
    }

    let data = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let nals = parse_annex_b(&data);

    // Warmup (page in data, warm caches/branch predictors, parse SPS/PPS paths)
    let frames = decode_once(&nals, threads);
    if frames == 0 {
        eprintln!("no frames decoded from {path}");
        std::process::exit(1);
    }

    let mut best = u128::MAX;
    let mut total = 0u128;
    for _ in 0..iterations {
        let t = Instant::now();
        let n = decode_once(&nals, threads);
        let elapsed = t.elapsed().as_micros();
        assert_eq!(n, frames, "frame count changed between passes");
        best = best.min(elapsed);
        total += elapsed;
    }

    let best_s = best as f64 / 1e6;
    let mean_s = total as f64 / 1e6 / iterations as f64;
    eprintln!(
        "{path}: {frames} frames | best {best_s:.3}s ({:.0} fps, {:.0} us/frame) | mean {mean_s:.3}s ({:.0} fps)",
        frames as f64 / best_s,
        best as f64 / frames as f64,
        frames as f64 / mean_s,
    );
    eprintln!(
        "threads: {threads}{}",
        if scalar { " (scalar forced)" } else { "" }
    );
}
