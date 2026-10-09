//! x86-64 SIMD deblocking filter (luma).
//!
//! Luma edges are filtered one 4-sample segment at a time with uniform
//! bs/alpha/beta/tc0 (see `deblock.rs` call sites), so one i32 lane per
//! sample-line covers a whole segment call: a vertical edge's four rows or
//! a horizontal edge's four columns. All math mirrors `deblock.rs`'s scalar
//! filter exactly — same rounding, same clamp order, per-lane `tc`
//! increments included.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::simd_x86::{level, SimdLevel};

/// Filter one luma edge segment: `s[lane]` holds 8 samples of one
/// sample-line, ordered `[p3, p2, p1, p0, q0, q1, q2, q3]`. For a vertical
/// edge a lane is a row; for a horizontal edge a lane is a column.
/// `bs` is the segment's boundary strength (1..=4; bs=0 segments are
/// skipped by the caller before dispatch).
///
/// NOT WIRED INTO deblock.rs: measured break-even. The 4-lane kernel beats
/// the scalar reference by ~25% but the per-segment gather/scatter needed
/// to feed it costs more than that (720p B-frames: 132 vs 116 fps through
/// the gathered paths, versus 159 fps for the original in-place scalar
/// loop). Kept as a bit-exact, differentially-tested building block for
/// the real design: 16-lane byte-domain kernels with in-register
/// transposes operating directly on the plane (FFmpeg-style), which
/// avoids the gather entirely.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub(crate) fn deblock_luma_segment(s: &mut [[u8; 8]; 4], bs: i32, alpha: i32, beta: i32, tc0: i32) {
    if level() == SimdLevel::Scalar {
        scalar_deblock_luma_segment(s, bs, alpha, beta, tc0);
        return;
    }
    unsafe { sse2_deblock_luma_segment(s, bs, alpha, beta, tc0) }
}

/// No-op on non-x86-64 (deblock.rs keeps its scalar path).
#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
pub(crate) fn deblock_luma_segment(_s: &mut [[u8; 8]; 4], _bs: i32, _alpha: i32, _beta: i32, _tc0: i32) {}

/// Scalar reference, mirroring deblock.rs's inner filter byte for byte.
/// Used as the non-x86-64 path, the forced-scalar path, and the
/// differential-test oracle.
pub(crate) fn scalar_deblock_luma_segment(
    s: &mut [[u8; 8]; 4],
    bs: i32,
    alpha: i32,
    beta: i32,
    tc0: i32,
) {
    for lane in s.iter_mut() {
        let (p3, p2, p1, p0) = (lane[0] as i32, lane[1] as i32, lane[2] as i32, lane[3] as i32);
        let (q0, q1, q2, q3) = (lane[4] as i32, lane[5] as i32, lane[6] as i32, lane[7] as i32);
        if !((p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta) {
            continue;
        }
        let ap = (p2 - p0).abs();
        let aq = (q2 - q0).abs();
        if bs == 4 {
            let small_gap = (p0 - q0).abs() < ((alpha >> 2) + 2);
            if small_gap && ap < beta {
                lane[3] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
                lane[2] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
                lane[1] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
            } else {
                lane[3] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            }
            if small_gap && aq < beta {
                lane[4] = ((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3) as u8;
                lane[5] = ((q2 + q1 + q0 + p0 + 2) >> 2) as u8;
                lane[6] = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
            } else {
                lane[4] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
            }
        } else {
            let avg = (p0 + q0 + 1) >> 1;
            let mut tc = tc0;
            if ap < beta {
                tc += 1;
                if tc0 != 0 {
                    lane[2] = (p1 + (((p2 + avg) >> 1) - p1).clamp(-tc0, tc0)) as u8;
                }
            }
            if aq < beta {
                tc += 1;
                if tc0 != 0 {
                    lane[5] = (q1 + (((q2 + avg) >> 1) - q1).clamp(-tc0, tc0)) as u8;
                }
            }
            let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            lane[3] = (p0 + delta).clamp(0, 255) as u8;
            lane[4] = (q0 - delta).clamp(0, 255) as u8;
        }
    }
}

/// SSE2 4-lane implementation. One i32 lane per sample-line; masks are
/// all-ones/all-zero per lane; selects via AND/ANDNOT (SSE2 has no blendv,
/// pabsd, or min/max_epi32, so abs/clamps are built from cmp+select).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline(never)]
unsafe fn sse2_deblock_luma_segment(
    s: &mut [[u8; 8]; 4],
    bs: i32,
    alpha: i32,
    beta: i32,
    tc0: i32,
) {
    let lane = |k: usize| {
        _mm_set_epi32(
            s[3][k] as i32,
            s[2][k] as i32,
            s[1][k] as i32,
            s[0][k] as i32,
        )
    };
    let p3 = lane(0);
    let p2 = lane(1);
    let p1 = lane(2);
    let p0 = lane(3);
    let q0 = lane(4);
    let q1 = lane(5);
    let q2 = lane(6);
    let q3 = lane(7);

    let zero = _mm_setzero_si128();
    let sel = |m: __m128i, a: __m128i, b: __m128i| -> __m128i {
        _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b))
    };
    let absd = |a: __m128i, b: __m128i| -> __m128i {
        let d = _mm_sub_epi32(a, b);
        let t = _mm_srai_epi32(d, 31);
        _mm_sub_epi32(_mm_xor_si128(d, t), t)
    };
    let lt = |a: __m128i, b: __m128i| _mm_cmplt_epi32(a, b);
    let cl = |x: __m128i, lo: __m128i, hi: __m128i| sel(lt(x, lo), lo, sel(lt(hi, x), hi, x));
    let two = |x: __m128i| _mm_add_epi32(x, x);
    let three = |x: __m128i| _mm_add_epi32(x, two(x));
    let f2 = |x: __m128i| _mm_srai_epi32(x, 2);
    let f3 = |x: __m128i| _mm_srai_epi32(x, 3);

    let alpha_v = _mm_set1_epi32(alpha);
    let beta_v = _mm_set1_epi32(beta);

    let gate = _mm_and_si128(
        _mm_and_si128(lt(absd(p0, q0), alpha_v), lt(absd(p1, p0), beta_v)),
        lt(absd(q1, q0), beta_v),
    );

    let ap = absd(p2, p0);
    let aq = absd(q2, q0);

    let mut np1 = p1;
    let mut nq1 = q1;
    let mut np2 = p2;
    let mut nq2 = q2;
    let np0;
    let nq0;
    if bs == 4 {
        let small_gap = lt(absd(p0, q0), _mm_set1_epi32((alpha >> 2) + 2));
        let pfull = _mm_and_si128(small_gap, lt(ap, beta_v));
        let qfull = _mm_and_si128(small_gap, lt(aq, beta_v));
        let c2 = _mm_set1_epi32(2);
        let c4 = _mm_set1_epi32(4);

        // strong p0' = (p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3
        let strong_p0 = f3(_mm_add_epi32(
            _mm_add_epi32(p2, two(_mm_add_epi32(p1, p0))),
            _mm_add_epi32(two(q0), _mm_add_epi32(q1, c4)),
        ));
        // weak p0' = (2*p1 + p0 + q1 + 2) >> 2
        let weak_p0 = f2(_mm_add_epi32(
            two(p1),
            _mm_add_epi32(p0, _mm_add_epi32(q1, c2)),
        ));
        np0 = sel(pfull, strong_p0, weak_p0);
        // strong p1' = (p2 + p1 + p0 + q0 + 2) >> 2
        np1 = sel(
            pfull,
            f2(_mm_add_epi32(
                _mm_add_epi32(p2, p1),
                _mm_add_epi32(p0, _mm_add_epi32(q0, c2)),
            )),
            p1,
        );
        // strong p2' = (2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3
        np2 = sel(
            pfull,
            f3(_mm_add_epi32(
                _mm_add_epi32(two(p3), _mm_add_epi32(three(p2), p1)),
                _mm_add_epi32(p0, _mm_add_epi32(q0, c4)),
            )),
            p2,
        );

        // q-side mirror
        let strong_q0 = f3(_mm_add_epi32(
            _mm_add_epi32(q2, two(_mm_add_epi32(q1, q0))),
            _mm_add_epi32(two(p0), _mm_add_epi32(p1, c4)),
        ));
        let weak_q0 = f2(_mm_add_epi32(
            two(q1),
            _mm_add_epi32(q0, _mm_add_epi32(p1, c2)),
        ));
        nq0 = sel(qfull, strong_q0, weak_q0);
        nq1 = sel(
            qfull,
            f2(_mm_add_epi32(
                _mm_add_epi32(q2, q1),
                _mm_add_epi32(q0, _mm_add_epi32(p0, c2)),
            )),
            q1,
        );
        nq2 = sel(
            qfull,
            f3(_mm_add_epi32(
                _mm_add_epi32(two(q3), _mm_add_epi32(three(q2), q1)),
                _mm_add_epi32(q0, _mm_add_epi32(p0, c4)),
            )),
            q2,
        );
    } else {
        let cond_p = lt(ap, beta_v);
        let cond_q = lt(aq, beta_v);
        let one = _mm_set1_epi32(1);
        // Per-lane tc = tc0 + (ap < beta) + (aq < beta), like the scalar
        // filter's side-effect increments.
        let tc = _mm_add_epi32(
            _mm_add_epi32(_mm_set1_epi32(tc0), _mm_and_si128(cond_p, one)),
            _mm_and_si128(cond_q, one),
        );
        let avg = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(p0, q0), one), 1);
        if tc0 != 0 {
            let tc0_v = _mm_set1_epi32(tc0);
            let ntc0 = _mm_sub_epi32(zero, tc0_v);
            np1 = sel(
                cond_p,
                _mm_add_epi32(
                    p1,
                    cl(
                        _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(p2, avg), 1), p1),
                        ntc0,
                        tc0_v,
                    ),
                ),
                p1,
            );
            nq1 = sel(
                cond_q,
                _mm_add_epi32(
                    q1,
                    cl(
                        _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(q2, avg), 1), q1),
                        ntc0,
                        tc0_v,
                    ),
                ),
                q1,
            );
        }
        let ntc = _mm_sub_epi32(zero, tc);
        let delta = cl(
            f3(_mm_add_epi32(
                _mm_add_epi32(_mm_slli_epi32(_mm_sub_epi32(q0, p0), 2), _mm_sub_epi32(p1, q1)),
                _mm_set1_epi32(4),
            )),
            ntc,
            tc,
        );
        let c255 = _mm_set1_epi32(255);
        np0 = cl(_mm_add_epi32(p0, delta), zero, c255);
        nq0 = cl(_mm_sub_epi32(q0, delta), zero, c255);
    }

    let mut store = |k: usize, v: __m128i| {
        let mut arr = [0i32; 4];
        _mm_storeu_si128(arr.as_mut_ptr() as *mut __m128i, v);
        s[0][k] = arr[0] as u8;
        s[1][k] = arr[1] as u8;
        s[2][k] = arr[2] as u8;
        s[3][k] = arr[3] as u8;
    };
    store(3, sel(gate, np0, p0));
    store(2, sel(gate, np1, p1));
    store(1, sel(gate, np2, p2));
    store(4, sel(gate, nq0, q0));
    store(5, sel(gate, nq1, q1));
    store(6, sel(gate, nq2, q2));
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    static PATH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Differential test: the SSE2 kernel must match the scalar reference
    /// byte for byte across boundary-strength/filter-parameter sweeps on
    /// biased-random samples (edges biased to exercise the gate and the
    /// strong/weak branch mixes).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn differential_deblock_luma_segment() {
        let _guard = PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // alpha/beta derived from QP-like ranges plus extremes; tc0 from
        // the TC0 table domain (0..=25) plus extremes.
        let mut rng = Rng(0x5eed_beef);
        let mut cases = 0u64;
        for _iter in 0..20000 {
            let mut s = [[0u8; 8]; 4];
            let edge = (rng.next_u32() % 8) as i32;
            for lane in s.iter_mut() {
                let base = rng.byte() as i32;
                for (k, v) in lane.iter_mut().enumerate() {
                    // Bias samples near a common base so gates actually pass.
                    *v = (base + (rng.next_u32() % 4) as i32 - edge * (k as i32 / 4)).clamp(0, 255) as u8;
                }
            }
            for bs in [1i32, 2, 3, 4] {
                for alpha in [0i32, 1, 2, 5, 9, 16, 24, 33, 51, 102, 255] {
                    for beta in [0i32, 2, 4, 8, 16, 31, 60, 255] {
                        for tc0 in [0i32, 1, 3, 7, 12, 18, 25, 40] {
                            let mut a = s;
                            let mut b = s;
                            crate::simd_deblock::scalar_deblock_luma_segment(&mut a, bs, alpha, beta, tc0);
                            crate::simd_x86::set_force_scalar(false);
                            crate::simd_deblock::deblock_luma_segment(&mut b, bs, alpha, beta, tc0);
                            crate::simd_x86::set_force_scalar(true);
                            for l in 0..4 {
                                for k in 0..8 {
                                    assert_eq!(
                                        a[l][k], b[l][k],
                                        "mismatch bs={bs} alpha={alpha} beta={beta} tc0={tc0} lane={l} idx={k}"
                                    );
                                }
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert!(cases > 0);
    }
}
