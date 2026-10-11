/// Benchmark decoder throughput on an H.264 Annex B bitstream.
///
/// Usage: cargo run --release --example bench_decode -- <input.h264> [--iterations N] [--scalar] [--threads N]
use rust_h264::decoder::Frame;
use rust_h264::nal::{parse_annex_b, NalUnit};
use std::time::Instant;

#[derive(Debug, PartialEq, Eq)]
struct DecodeResult {
    frames: usize,
    fingerprint: u64,
}

fn add_frame(hash: &mut u64, frame: &Frame) {
    for byte in frame
        .width
        .to_le_bytes()
        .into_iter()
        .chain(frame.height.to_le_bytes())
        .chain(frame.pic_order_cnt.to_le_bytes())
        .chain(frame.y.iter().copied())
        .chain(frame.u.iter().copied())
        .chain(frame.v.iter().copied())
    {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
}

// `verify` is false on timed passes, preserving the streaming decode/drop workload.
fn decode_once(nals: &[NalUnit], threads: usize, verify: bool) -> Result<DecodeResult, String> {
    let mut result = DecodeResult {
        frames: 0,
        fingerprint: 0xcbf29ce484222325,
    };
    let mut record = |frames: Vec<Frame>| {
        result.frames += frames.len();
        if verify {
            for frame in &frames {
                add_frame(&mut result.fingerprint, frame);
            }
        }
    };
    if threads <= 1 {
        let mut decoder = rust_h264::decoder::OrderedDecoder::new();
        for (index, nal) in nals.iter().enumerate() {
            record(
                decoder
                    .decode_nal(nal)
                    .map_err(|e| format!("NAL {index}: {e}"))?,
            );
        }
        record(decoder.flush());
    } else {
        let mut decoder = rust_h264::threading::ThreadedDecoder::new(threads);
        for (index, nal) in nals.iter().enumerate() {
            record(
                decoder
                    .decode_nal(nal)
                    .map_err(|e| format!("NAL {index}: {e}"))?,
            );
        }
        record(decoder.flush());
    }
    Ok(result)
}

fn positive_usize(value: Option<&String>, flag: &str) -> Result<usize, String> {
    let parsed = value
        .ok_or_else(|| format!("missing value for {flag}"))?
        .parse::<usize>()
        .map_err(|_| format!("invalid value for {flag}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(parsed)
}

fn median_us(times: &[u128]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
    } else {
        sorted[mid] as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path: Option<&str> = None;
    let mut iterations = 3usize;
    let mut scalar = false;
    let mut threads = 1usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" => {
                iterations = positive_usize(args.get(i + 1), "--iterations")
                    .unwrap_or_else(|e| panic!("{e}"));
                i += 1;
            }
            "--scalar" => scalar = true,
            "--threads" => {
                threads =
                    positive_usize(args.get(i + 1), "--threads").unwrap_or_else(|e| panic!("{e}"));
                i += 1;
            }
            other if other.starts_with('-') => panic!("unknown option: {other}"),
            other => {
                if path.replace(other).is_some() {
                    panic!("multiple input paths supplied");
                }
            }
        }
        i += 1;
    }
    let path = path.unwrap_or_else(|| {
        eprintln!("Usage: bench_decode <input.h264> [--iterations N] [--scalar] [--threads N]");
        std::process::exit(1);
    });

    #[cfg(target_arch = "x86_64")]
    if scalar {
        rust_h264::set_force_scalar(true);
    }

    let data = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let nals = parse_annex_b(&data);

    // Pixel checks run outside every timed interval, including warmup.
    let warmup =
        decode_once(&nals, threads, true).unwrap_or_else(|e| panic!("warmup decode failed: {e}"));
    if warmup.frames == 0 {
        panic!("no frames decoded from {path}");
    }
    eprintln!("warmup fingerprint: {:016x}", warmup.fingerprint);

    let mut times = Vec::with_capacity(iterations);
    let mut best = u128::MAX;
    let mut best_deblock_ns = 0u64;
    for pass in 0..iterations {
        let deblock_before = rust_h264::deblock_ns();
        let start = Instant::now();
        let decoded = decode_once(&nals, threads, false)
            .unwrap_or_else(|e| panic!("pass {} decode failed: {e}", pass + 1));
        let elapsed = start.elapsed().as_micros();
        let deblock_ns = rust_h264::deblock_ns().saturating_sub(deblock_before);
        assert_eq!(
            decoded.frames,
            warmup.frames,
            "frame count changed on pass {}",
            pass + 1
        );
        if elapsed < best {
            best = elapsed;
            best_deblock_ns = deblock_ns;
        }
        times.push(elapsed);

        let checked = decode_once(&nals, threads, true)
            .unwrap_or_else(|e| panic!("validation after pass {} failed: {e}", pass + 1));
        assert_eq!(
            checked,
            warmup,
            "decoded output changed after pass {}",
            pass + 1
        );
    }

    let frames = warmup.frames;
    let best_s = best as f64 / 1e6;
    let median_s = median_us(&times) / 1e6;
    let mean_s = times.iter().sum::<u128>() as f64 / 1e6 / iterations as f64;
    eprintln!(
        "{path}: {frames} frames | best {best_s:.3}s ({:.0} fps, {:.0} us/frame) | median {median_s:.3}s ({:.0} fps) | mean {mean_s:.3}s ({:.0} fps)",
        frames as f64 / best_s, best as f64 / frames as f64,
        frames as f64 / median_s, frames as f64 / mean_s,
    );
    eprintln!(
        "threads: {threads}{}",
        if scalar { " (scalar forced)" } else { "" }
    );
    let deblock_us_per_frame = best_deblock_ns as f64 / 1000.0 / frames as f64;
    let wall_us_per_frame = best as f64 / frames as f64;
    if threads == 1 {
        eprintln!(
            "deblock share of best pass: {:.1}% ({deblock_us_per_frame:.0} us/frame of {wall_us_per_frame:.0} us/frame)",
            100.0 * best_deblock_ns as f64 / (best as f64 * 1000.0),
        );
    } else {
        eprintln!(
            "best pass: accumulated deblock worker time {deblock_us_per_frame:.0} us/frame; wall time {wall_us_per_frame:.0} us/frame",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{median_us, positive_usize};

    #[test]
    fn arguments_reject_invalid_counts() {
        assert!(positive_usize(None, "--iterations").is_err());
        assert!(positive_usize(Some(&"0".to_string()), "--iterations").is_err());
        assert!(positive_usize(Some(&"bad".to_string()), "--iterations").is_err());
        assert_eq!(
            positive_usize(Some(&"2".to_string()), "--iterations"),
            Ok(2)
        );
    }

    #[test]
    fn median_handles_odd_and_even_passes() {
        assert_eq!(median_us(&[9, 3, 5]), 5.0);
        assert_eq!(median_us(&[9, 3, 5, 7]), 6.0);
    }
}
