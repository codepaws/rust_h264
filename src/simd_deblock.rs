//! x86-64 SIMD deblocking filter (luma).
//!
//! Two implementations live here:
//!
//! - [`deblock_luma_v16`] / [`deblock_luma_h16`]: 16-lane byte-domain
//!   whole-edge kernels operating directly on the plane — no gather; the
//!   vertical variant uses in-register transposes (FFmpeg-style), the
//!   horizontal variant loads row vectors directly. **Dormant**: with
//!   movemask gate early-outs and per-bs path guards they measure parity
//!   with the scalar loop (4-round averages: -3.2% at 720p, +1.4% at
//!   1080p), because on gate-dense content the branchy scalar loop's
//!   per-row exits cost less than the branchless vector paths. A real win
//!   needs FFmpeg-level specialization (dedicated per-bs kernels, paired
//!   edges) rather than generic branchless vectorization.
//! - [`deblock_luma_segment`]: a dormant 4-lane (i32 lane per
//!   sample-line) kernel over a gathered `[[u8; 8]; 4]` block. Bit-exact
//!   and differentially tested, but an in-place gather/scatter
//!   integration measured -6..9% vs the scalar loop, so it is not wired
//!   in; kept as a building block.
//!
//! All math mirrors `deblock.rs`'s scalar filter exactly — same rounding,
//! same clamp order, per-lane `tc` increments included. The whole-edge
//! kernels reproduce the per-segment scalar path byte for byte, verified
//! by differential tests and the full byte-exact corpus.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::simd_x86::{level, SimdLevel};

// ---------------------------------------------------------------------------
// Whole-edge 16-lane kernels (wired into deblock.rs)
// ---------------------------------------------------------------------------

/// Filter one full vertical luma edge: 16 rows starting at `(x, y)`, edge
/// between columns `x-1` and `x`. `bs`/`tc0` are per 4-row segment.
/// Returns `false` when the SIMD path does not apply (non-x86-64, scalar
/// forced, or the sample window is out of bounds) — callers fall back to
/// the per-segment scalar filter.
///
/// **Dormant** — see module docs (measured parity with the scalar loop).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn deblock_luma_v16(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    alpha: i32,
    beta: i32,
    bs: [i32; 4],
    tc0: [i32; 4],
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if level() != SimdLevel::Scalar
            && x >= 4
            && (y + 15) * stride + x + 3 < plane.len()
            && bs.iter().all(|&b| (0..=4).contains(&b))
            && (0..=127).contains(&alpha)
            && (0..=127).contains(&beta)
        {
            unsafe {
                sse2_deblock_luma_v16(plane, stride, x, y, alpha as u8, beta as u8, bs, tc0);
            }
            return true;
        }
    }
    false
}

/// Filter one full horizontal luma edge: 16 columns starting at `x`, edge
/// between rows `y-1` and `y`. `bs`/`tc0` are per 4-column segment.
/// Returns `false` when falling back is required.
///
/// **Dormant** — see module docs (measured parity with the scalar loop).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn deblock_luma_h16(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    alpha: i32,
    beta: i32,
    bs: [i32; 4],
    tc0: [i32; 4],
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if level() != SimdLevel::Scalar
            && y >= 4
            && (y + 3) * stride + x + 15 < plane.len()
            && bs.iter().all(|&b| (0..=4).contains(&b))
            && (0..=127).contains(&alpha)
            && (0..=127).contains(&beta)
        {
            unsafe {
                sse2_deblock_luma_h16(plane, stride, x, y, alpha as u8, beta as u8, bs, tc0);
            }
            return true;
        }
    }
    false
}

/// Cheap prelude check: true when at least one lane has bs > 0 and passes
/// the spec 8.7.2.3 filtering condition. Used to skip the full filter math
/// and the scatter for fully-inactive edges (the common case).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code, non_snake_case)]
unsafe fn gate_any(
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    alpha: u8,
    beta: u8,
    bs_v: __m128i,
) -> bool {
    let zero = _mm_setzero_si128();
    let alpha_v = _mm_set1_epi8(alpha as i8);
    let beta_v = _mm_set1_epi8(beta as i8);
    let absd = |a: __m128i, b: __m128i| _mm_adds_epu8(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    let lt = |d: __m128i, t: __m128i| {
        let le = _mm_cmpeq_epi8(_mm_subs_epu8(d, t), zero);
        let eq = _mm_cmpeq_epi8(d, t);
        _mm_andnot_si128(eq, le) // le & !eq = strict <
    };
    let gate = _mm_and_si128(
        _mm_and_si128(lt(absd(p0, q0), alpha_v), lt(absd(p1, p0), beta_v)),
        lt(absd(q1, q0), beta_v),
    );
    let active_bs = _mm_cmpgt_epi8(bs_v, zero);
    let m = _mm_and_si128(gate, active_bs);
    _mm_movemask_epi8(m) != 0
}

/// Per-4-lane `bs`/`tc0` broadcast for `lanes` (8 or 16) lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn seg_vec(vals: [i32; 4], lanes: usize, offset: usize) -> __m128i {
    let mut arr = [0u8; 16];
    for (i, slot) in arr.iter_mut().enumerate().take(lanes) {
        *slot = vals[offset + i / 4] as u8;
    }
    unsafe { _mm_loadu_si128(arr.as_ptr() as *const __m128i) }
}

/// Byte-domain core filter over one vector of sample lines. Lane-
/// independently equivalent to the scalar segment filter, including
/// per-lane `tc` increments and the `tc0 != 0` guard.
///
/// Sample vectors are the p/q columns (vertical edge) or rows (horizontal
/// edge); `alpha`/`beta` are the edge-wide filter parameters.
///
/// Returns `(np2, np1, np0, nq0, nq1, nq2)`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
unsafe fn deblock_core(
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    p3: __m128i,
    q3: __m128i,
    alpha: u8,
    beta: u8,
    bs_v: __m128i,
    tc0_v: __m128i,
    tc0_nonzero: bool,
    has_bs4: bool,
    has_normal: bool,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi8(1);
    let ff = _mm_set1_epi8(-1);
    let alpha_v = _mm_set1_epi8(alpha as i8);
    let beta_v = _mm_set1_epi8(beta as i8);

    let and = _mm_and_si128;
    let or = _mm_or_si128;
    let xor = _mm_xor_si128;
    let sel = |m: __m128i, a: __m128i, b: __m128i| or(and(m, a), _mm_andnot_si128(m, b));
    let absd = |a: __m128i, b: __m128i| _mm_adds_epu8(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    // Strict unsigned byte `<`: saturating d-t == 0 means d <= t, so also
    // require d != t. (alpha/beta thresholds stay <= 127; the abs diffs
    // compared against them can be up to 255, handled by the saturating
    // sub rather than a signed compare.)
    let lt = |d: __m128i, t: __m128i| {
        and(
            _mm_cmpeq_epi8(_mm_subs_epu8(d, t), zero),
            xor(_mm_cmpeq_epi8(d, t), ff),
        )
    };

    // Gate (spec 8.7.2.3 filtering condition).
    let gate = and(
        and(lt(absd(p0, q0), alpha_v), lt(absd(p1, p0), beta_v)),
        lt(absd(q1, q0), beta_v),
    );

    let ap = absd(p2, p0);
    let aq = absd(q2, q0);
    let cond_p = lt(ap, beta_v);
    let cond_q = lt(aq, beta_v);

    let bs4 = _mm_cmpeq_epi8(bs_v, _mm_set1_epi8(4));
    let bsn = and(_mm_cmpgt_epi8(bs_v, zero), _mm_xor_si128(bs4, ff));

    // i16 halves helpers for the signed/summing paths.
    let lo16 = |v: __m128i| _mm_unpacklo_epi8(v, zero);
    let hi16 = |v: __m128i| _mm_unpackhi_epi8(v, zero);
    let pack_sat = _mm_packus_epi16;
    let clamp_i16 = |x: __m128i, lim: __m128i| {
        let nlim = _mm_sub_epi16(zero, lim);
        _mm_min_epi16(_mm_max_epi16(x, nlim), lim)
    };

    // ---- normal filter (bs 1..3) ----
    // Per-lane tc = tc0 + (ap < beta) + (aq < beta).
    let tc = _mm_adds_epu8(
        _mm_adds_epu8(tc0_v, and(cond_p, one)),
        and(cond_q, one),
    );
    let avg = _mm_avg_epu8(p0, q0); // (p0 + q0 + 1) >> 1, same rounding as scalar

    // p1' = p1 + clamp(((p2 + avg) >> 1) - p1, ±tc0) when ap < beta.
    let mut np1 = p1;
    let mut nq1 = q1;
    let mut np0_normal = p0;
    let mut nq0_normal = q0;
    if has_normal && tc0_nonzero {
        let tp_lo = _mm_srai_epi16(_mm_add_epi16(lo16(p2), lo16(avg)), 1);
        let tp_hi = _mm_srai_epi16(_mm_add_epi16(hi16(p2), hi16(avg)), 1);
        let tq_lo = _mm_srai_epi16(_mm_add_epi16(lo16(q2), lo16(avg)), 1);
        let tq_hi = _mm_srai_epi16(_mm_add_epi16(hi16(q2), hi16(avg)), 1);
        let t0_lo = lo16(tc0_v);
        let t0_hi = hi16(tc0_v);
        let dp_lo = clamp_i16(_mm_sub_epi16(tp_lo, lo16(p1)), t0_lo);
        let dp_hi = clamp_i16(_mm_sub_epi16(tp_hi, hi16(p1)), t0_hi);
        let dq_lo = clamp_i16(_mm_sub_epi16(tq_lo, lo16(q1)), t0_lo);
        let dq_hi = clamp_i16(_mm_sub_epi16(tq_hi, hi16(q1)), t0_hi);
        np1 = sel(
            cond_p,
            pack_sat(_mm_add_epi16(lo16(p1), dp_lo), _mm_add_epi16(hi16(p1), dp_hi)),
            p1,
        );
        nq1 = sel(
            cond_q,
            pack_sat(_mm_add_epi16(lo16(q1), dq_lo), _mm_add_epi16(hi16(q1), dq_hi)),
            q1,
        );
    }

    // delta = clamp((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3, ±tc);
    // p0' = clamp(p0 + delta), q0' = clamp(q0 - delta) (packus = clamp 0..255).
    let delta_lane = |q0l: __m128i, p0l: __m128i, p1l: __m128i, q1l: __m128i, tcl: __m128i| {
        let d = _mm_slli_epi16(_mm_sub_epi16(q0l, p0l), 2);
        let d = _mm_add_epi16(d, _mm_sub_epi16(p1l, q1l));
        let d = _mm_add_epi16(d, _mm_set1_epi16(4));
        clamp_i16(_mm_srai_epi16(d, 3), tcl)
    };
    if has_normal {
        let d_lo = delta_lane(lo16(q0), lo16(p0), lo16(p1), lo16(q1), lo16(tc));
        let d_hi = delta_lane(hi16(q0), hi16(p0), hi16(p1), hi16(q1), hi16(tc));
        np0_normal = pack_sat(_mm_add_epi16(lo16(p0), d_lo), _mm_add_epi16(hi16(p0), d_hi));
        nq0_normal = pack_sat(_mm_sub_epi16(lo16(q0), d_lo), _mm_sub_epi16(hi16(q0), d_hi));
    }

    // ---- strong filter (bs == 4) ----
    let mut np0_strong = p0;
    let mut np1_strong = p1;
    let mut np2_strong = p2;
    let mut nq0_strong = q0;
    let mut nq1_strong = q1;
    let mut nq2_strong = q2;
    if has_bs4 {
    let small_t = _mm_set1_epi8(((alpha >> 2) + 2) as i8);
    let small_gap = lt(absd(p0, q0), small_t);
    let pfull = and(small_gap, cond_p);
    let qfull = and(small_gap, cond_q);

    let two16 = |v: __m128i| _mm_add_epi16(v, v);
    let three16 = |v: __m128i| _mm_add_epi16(v, two16(v));
    let c2 = _mm_set1_epi16(2);
    let c4 = _mm_set1_epi16(4);

    // Strong/weak outputs need u16 headroom (sums up to ~2054); values are
    // provably in [0, 255] after the shift, so packus matches the scalar
    // `as u8` exactly.
    let sp0 = |L: &dyn Fn(__m128i) -> __m128i| {
        let s = _mm_add_epi16(
            _mm_add_epi16(L(p2), two16(_mm_add_epi16(L(p1), L(p0)))),
            _mm_add_epi16(two16(L(q0)), _mm_add_epi16(L(q1), c4)),
        );
        _mm_srai_epi16(s, 3)
    };
    let sp1 = |L: &dyn Fn(__m128i) -> __m128i| {
        let s = _mm_add_epi16(
            _mm_add_epi16(L(p2), L(p1)),
            _mm_add_epi16(L(p0), _mm_add_epi16(L(q0), c2)),
        );
        _mm_srai_epi16(s, 2)
    };
    let sp2 = |L: &dyn Fn(__m128i) -> __m128i| {
        let s = _mm_add_epi16(
            _mm_add_epi16(two16(L(p3)), _mm_add_epi16(three16(L(p2)), L(p1))),
            _mm_add_epi16(L(p0), _mm_add_epi16(L(q0), c4)),
        );
        _mm_srai_epi16(s, 3)
    };
    let sq0 = |L: &dyn Fn(__m128i) -> __m128i| {
        let s = _mm_add_epi16(
            _mm_add_epi16(L(q2), two16(_mm_add_epi16(L(q1), L(q0)))),
            _mm_add_epi16(two16(L(p0)), _mm_add_epi16(L(p1), c4)),
        );
        _mm_srai_epi16(s, 3)
    };
    let sq1 = |L: &dyn Fn(__m128i) -> __m128i| {
        let s = _mm_add_epi16(
            _mm_add_epi16(L(q2), L(q1)),
            _mm_add_epi16(L(q0), _mm_add_epi16(L(p0), c2)),
        );
        _mm_srai_epi16(s, 2)
    };
    let sq2 = |L: &dyn Fn(__m128i) -> __m128i| {
        let s = _mm_add_epi16(
            _mm_add_epi16(two16(L(q3)), _mm_add_epi16(three16(L(q2)), L(q1))),
            _mm_add_epi16(L(q0), _mm_add_epi16(L(p0), c4)),
        );
        _mm_srai_epi16(s, 3)
    };
    let wp0 = |L: &dyn Fn(__m128i) -> __m128i| {
        let s = _mm_add_epi16(two16(L(p1)), _mm_add_epi16(L(p0), _mm_add_epi16(L(q1), c2)));
        _mm_srai_epi16(s, 2)
    };
    let wq0 = |L: &dyn Fn(__m128i) -> __m128i| {
        let s = _mm_add_epi16(two16(L(q1)), _mm_add_epi16(L(q0), _mm_add_epi16(L(p1), c2)));
        _mm_srai_epi16(s, 2)
    };

    np0_strong = sel(
        pfull,
        pack_sat(sp0(&lo16), sp0(&hi16)),
        pack_sat(wp0(&lo16), wp0(&hi16)),
    );
    np1_strong = sel(pfull, pack_sat(sp1(&lo16), sp1(&hi16)), p1);
    np2_strong = sel(pfull, pack_sat(sp2(&lo16), sp2(&hi16)), p2);
    nq0_strong = sel(
        qfull,
        pack_sat(sq0(&lo16), sq0(&hi16)),
        pack_sat(wq0(&lo16), wq0(&hi16)),
    );
    nq1_strong = sel(qfull, pack_sat(sq1(&lo16), sq1(&hi16)), q1);
    nq2_strong = sel(qfull, pack_sat(sq2(&lo16), sq2(&hi16)), q2);
    }

    // ---- select strong / normal / identity per lane, then apply gate ----
    let active = and(gate, or(bs4, bsn));
    let pick = |strong: __m128i, normal: __m128i, orig: __m128i| {
        sel(active, sel(bs4, strong, normal), orig)
    };
    (
        pick(np2_strong, p2, p2), // p2' only changes in the strong path
        pick(np1_strong, np1, p1),
        pick(np0_strong, np0_normal, p0),
        pick(nq0_strong, nq0_normal, q0),
        pick(nq1_strong, nq1, q1),
        pick(nq2_strong, q2, q2),
    )
}

/// Vertical edge: gather columns via in-register transposes, filter,
/// scatter. The filter is lane-independent, so the edge is processed as
/// two 8-row halves with no combining.
///
/// Forward transpose (rows -> columns): pair rows with unpacklo_epi8,
/// then unpacklo_epi16 / unpacklo_epi32. After the three levels each
/// intermediate holds two 8-lane column vectors in its low/high 64 bits.
/// The inverse is the same unpack_epi64 chain run backwards, ending with
/// two rows per register.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn sse2_deblock_luma_v16(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    alpha: u8,
    beta: u8,
    bs: [i32; 4],
    tc0: [i32; 4],
) {
    let tc0_nonzero = tc0.iter().any(|&t| t != 0);

    // One level-3 unpack chain = an 8x8 byte transpose: the eight input
    // vectors (a "row" in the low 8 bytes of each) become four output
    // vectors holding two 8-lane "columns" in their low/high 64 bits.
    // It is used in both directions: rows -> columns before filtering and
    // columns -> rows after (a transpose of the transpose).
    let transpose8 = |r: [__m128i; 8]| -> [__m128i; 4] {
        let d0 = _mm_unpacklo_epi8(r[0], r[1]);
        let d1 = _mm_unpacklo_epi8(r[2], r[3]);
        let d2 = _mm_unpacklo_epi8(r[4], r[5]);
        let d3 = _mm_unpacklo_epi8(r[6], r[7]);
        let f0 = _mm_unpacklo_epi16(d0, d1);
        let f1 = _mm_unpackhi_epi16(d0, d1);
        let f2 = _mm_unpacklo_epi16(d2, d3);
        let f3 = _mm_unpackhi_epi16(d2, d3);
        [
            _mm_unpacklo_epi32(f0, f2),
            _mm_unpackhi_epi32(f0, f2),
            _mm_unpacklo_epi32(f1, f3),
            _mm_unpackhi_epi32(f1, f3),
        ]
    };
    let col_pair = |h: __m128i| {
        (
            _mm_unpacklo_epi64(h, _mm_setzero_si128()),
            _mm_unpackhi_epi64(h, _mm_setzero_si128()),
        )
    };

    for half in 0..2 {
        let y0 = y + half * 8;
        // 8 rows x 8 bytes: [p3, p2, p1, p0, q0, q1, q2, q3].
        let mut rows = [_mm_setzero_si128(); 8];
        for (i, rv) in rows.iter_mut().enumerate() {
            *rv = _mm_loadl_epi64(plane.as_ptr().add((y0 + i) * stride + x - 4) as *const __m128i);
        }
        let h = transpose8(rows);
        let (p3, p2) = col_pair(h[0]); // columns x-4, x-3
        let (p1, p0) = col_pair(h[1]); // columns x-2, x-1
        let (q0, q1) = col_pair(h[2]); // columns x,   x+1
        let (q2, q3) = col_pair(h[3]); // columns x+2, x+3

        let seg = half * 2;
        let bs_half = [bs[seg], bs[seg + 1], 0, 0];
        let bs_v = seg_vec(bs, 8, seg);
        let tc0_v = seg_vec(tc0, 8, seg);

        // Early-out: when no lane passes the gate (and has bs > 0), skip
        // the filter math and the scatter entirely — the common case on
        // smooth content, and the reason a naive always-compute kernel
        // loses to the branchy scalar loop.
        if !gate_any(p1, p0, q0, q1, alpha, beta, bs_v) {
            continue;
        }

        let (np2, np1, np0, nq0, nq1, nq2) = deblock_core(
            p2, p1, p0, q0, q1, q2, p3, q3, alpha, beta, bs_v, tc0_v, tc0_nonzero,
            bs_half.contains(&4),
            bs_half.iter().any(|&b| (1..4).contains(&b)),
        );

        // Transpose the (modified) columns back into rows.
        let hs = transpose8([p3, np2, np1, np0, nq0, nq1, nq2, q3]);
        for (i, hv) in hs.iter().enumerate() {
            store8(plane, (y0 + i * 2) * stride + x - 4, *hv);
            store8_hi(plane, (y0 + i * 2 + 1) * stride + x - 4, *hv);
        }
    }
}

/// Horizontal edge: samples are contiguous per row — direct 16-byte loads.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn sse2_deblock_luma_h16(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    alpha: u8,
    beta: u8,
    bs: [i32; 4],
    tc0: [i32; 4],
) {
    let row = |dy: isize| {
        _mm_loadu_si128(
            plane.as_ptr().add(((y as isize + dy) as usize) * stride + x) as *const __m128i,
        )
    };
    let p3 = row(-4);
    let p2 = row(-3);
    let p1 = row(-2);
    let p0 = row(-1);
    let q0 = row(0);
    let q1 = row(1);
    let q2 = row(2);
    let q3 = row(3);

    let bs_v = seg_vec(bs, 16, 0);
    let tc0_v = seg_vec(tc0, 16, 0);
    let tc0_nonzero = tc0.iter().any(|&t| t != 0);

    if !gate_any(p1, p0, q0, q1, alpha, beta, bs_v) {
        return;
    }

    let (np2, np1, np0, nq0, nq1, nq2) = deblock_core(
        p2, p1, p0, q0, q1, q2, p3, q3, alpha, beta, bs_v, tc0_v, tc0_nonzero,
        bs.contains(&4),
        bs.iter().any(|&b| (1..4).contains(&b)),
    );

    let mut store = |dy: isize, v: __m128i| {
        let base = ((y as isize + dy) as usize) * stride + x;
        _mm_storeu_si128(plane.as_mut_ptr().add(base) as *mut __m128i, v);
    };
    store(-3, np2);
    store(-2, np1);
    store(-1, np0);
    store(0, nq0);
    store(1, nq1);
    store(2, nq2);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn store8(plane: &mut [u8], base: usize, v: __m128i) {
    _mm_storel_epi64(plane.as_mut_ptr().add(base) as *mut __m128i, v);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn store8_hi(plane: &mut [u8], base: usize, v: __m128i) {
    let hi = _mm_unpackhi_epi64(v, v);
    _mm_storel_epi64(plane.as_mut_ptr().add(base) as *mut __m128i, hi);
}

// ---------------------------------------------------------------------------
// Dormant 4-lane segment kernel (kept as documented building block)
// ---------------------------------------------------------------------------

/// Filter one luma edge segment: `s[lane]` holds 8 samples of one
/// sample-line, ordered `[p3, p2, p1, p0, q0, q1, q2, q3]`.
/// **Dormant** — see module docs.
#[allow(dead_code)]
pub(crate) fn deblock_luma_segment(s: &mut [[u8; 8]; 4], bs: i32, alpha: i32, beta: i32, tc0: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        if level() != SimdLevel::Scalar {
            unsafe { sse2_deblock_luma_segment(s, bs, alpha, beta, tc0) };
            return;
        }
    }
    scalar_deblock_luma_segment(s, bs, alpha, beta, tc0);
}

/// Scalar reference for one luma edge segment, mirroring deblock.rs's
/// inner filter byte for byte. Also the differential-test oracle for the
/// whole-edge kernels (applied per 4-lane segment group).
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

/// SSE2 4-lane implementation (dormant; see module docs).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline(never)]
#[allow(non_snake_case)]
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

        let strong_p0 = f3(_mm_add_epi32(
            _mm_add_epi32(p2, two(_mm_add_epi32(p1, p0))),
            _mm_add_epi32(two(q0), _mm_add_epi32(q1, c4)),
        ));
        let weak_p0 = f2(_mm_add_epi32(
            two(p1),
            _mm_add_epi32(p0, _mm_add_epi32(q1, c2)),
        ));
        np0 = sel(pfull, strong_p0, weak_p0);
        np1 = sel(
            pfull,
            f2(_mm_add_epi32(
                _mm_add_epi32(p2, p1),
                _mm_add_epi32(p0, _mm_add_epi32(q0, c2)),
            )),
            p1,
        );
        np2 = sel(
            pfull,
            f3(_mm_add_epi32(
                _mm_add_epi32(two(p3), _mm_add_epi32(three(p2), p1)),
                _mm_add_epi32(p0, _mm_add_epi32(q0, c4)),
            )),
            p2,
        );

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

    /// The whole-edge kernels must equal the per-segment scalar reference
    /// applied to the same samples. A 32x32 plane with a moderate gradient
    /// keeps the filtering gate passing for a realistic mix of lanes.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn differential_deblock_v16_h16() {
        let _guard = PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut rng = Rng(0x5eed_1337);
        let mut cases = 0u64;
        for _iter in 0..4000 {
            let mut plane = [0u8; 32 * 32];
            let base = rng.byte() as i32;
            for v in plane.iter_mut() {
                *v = (base + (rng.next_u32() % 6) as i32 - 2).clamp(0, 255) as u8;
            }
            let mut bs = [0i32; 4];
            let mut tc0 = [0i32; 4];
            for s in 0..4 {
                bs[s] = (rng.next_u32() % 5) as i32;
                tc0[s] = if bs[s] == 0 {
                    0
                } else {
                    (rng.next_u32() % 26) as i32
                };
            }
            for alpha in [1i32, 4, 9, 16, 24, 33, 51, 102] {
                for beta in [0i32, 2, 4, 8, 16, 31, 63] {
                    // Vertical: edge at column 12, rows 8..24.
                    let mut a = plane;
                    let mut b = plane;
                    for seg in 0..4 {
                        if bs[seg] == 0 {
                            continue; // deblock.rs skips bs=0 segments entirely
                        }
                        let mut blk = [[0u8; 8]; 4];
                        for (r, row) in blk.iter_mut().enumerate() {
                            let yy = 8 + seg * 4 + r;
                            row.copy_from_slice(&a[yy * 32 + 8..yy * 32 + 16]);
                        }
                        scalar_deblock_luma_segment(&mut blk, bs[seg], alpha, beta, tc0[seg]);
                        for (r, row) in blk.iter().enumerate() {
                            let yy = 8 + seg * 4 + r;
                            a[yy * 32 + 8..yy * 32 + 16].copy_from_slice(row);
                        }
                    }
                    crate::simd_x86::set_force_scalar(false);
                    let handled = deblock_luma_v16(&mut b, 32, 12, 8, alpha, beta, bs, tc0);
                    crate::simd_x86::set_force_scalar(true);
                    assert!(handled);
                    for (i, (&xa, &xb)) in a.iter().zip(b.iter()).enumerate() {
                        assert_eq!(
                            xa, xb,
                            "v16 mismatch at {i} alpha={alpha} beta={beta} bs={bs:?} tc0={tc0:?}"
                        );
                    }

                    // Horizontal: edge at row 12, columns 8..24.
                    let mut a = plane;
                    let mut b = plane;
                    for seg in 0..4 {
                        if bs[seg] == 0 {
                            continue;
                        }
                        let mut blk = [[0u8; 8]; 4];
                        for (c, col) in blk.iter_mut().enumerate() {
                            let xx = 8 + seg * 4 + c;
                            for (k, v) in col.iter_mut().enumerate() {
                                *v = a[(8 + k) * 32 + xx];
                            }
                        }
                        scalar_deblock_luma_segment(&mut blk, bs[seg], alpha, beta, tc0[seg]);
                        for (c, col) in blk.iter().enumerate() {
                            let xx = 8 + seg * 4 + c;
                            for (k, &v) in col.iter().enumerate() {
                                a[(8 + k) * 32 + xx] = v;
                            }
                        }
                    }
                    crate::simd_x86::set_force_scalar(false);
                    let handled = deblock_luma_h16(&mut b, 32, 8, 12, alpha, beta, bs, tc0);
                    crate::simd_x86::set_force_scalar(true);
                    assert!(handled);
                    for (i, (&xa, &xb)) in a.iter().zip(b.iter()).enumerate() {
                        assert_eq!(
                            xa, xb,
                            "h16 mismatch at {i} alpha={alpha} beta={beta} bs={bs:?} tc0={tc0:?}"
                        );
                    }
                    cases += 1;
                }
            }
        }
        assert!(cases > 0);
    }

    /// bs=0 everywhere must be a byte-identical no-op — isolates the
    /// transpose/scatter mapping from the filter math.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn v16_identity_bs0() {
        let _guard = PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut plane = [0u8; 32 * 32];
        for (i, v) in plane.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let mut b = plane;
        crate::simd_x86::set_force_scalar(false);
        let ok = deblock_luma_v16(&mut b, 32, 12, 8, 16, 16, [0, 0, 0, 0], [0, 0, 0, 0]);
        crate::simd_x86::set_force_scalar(true);
        assert!(ok);
        let bad: Vec<usize> = (0..plane.len()).filter(|&i| plane[i] != b[i]).collect();
        assert!(bad.is_empty(), "identity failed at {bad:?}");
        let mut b2 = plane;
        crate::simd_x86::set_force_scalar(false);
        let ok2 = deblock_luma_h16(&mut b2, 32, 8, 12, 16, 16, [0, 0, 0, 0], [0, 0, 0, 0]);
        crate::simd_x86::set_force_scalar(true);
        assert!(ok2);
        let bad2: Vec<usize> = (0..plane.len()).filter(|&i| plane[i] != b2[i]).collect();
        assert!(bad2.is_empty(), "h identity failed at {bad2:?}");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn differential_deblock_luma_segment() {
        let _guard = PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut rng = Rng(0x5eed_beef);
        for _iter in 0..20000 {
            let mut s = [[0u8; 8]; 4];
            let edge = (rng.next_u32() % 8) as i32;
            for lane in s.iter_mut() {
                let base = rng.byte() as i32;
                for (k, v) in lane.iter_mut().enumerate() {
                    *v = (base + (rng.next_u32() % 4) as i32 - edge * (k as i32 / 4))
                        .clamp(0, 255) as u8;
                }
            }
            for bs in [1i32, 2, 3, 4] {
                for alpha in [0i32, 1, 2, 5, 9, 16, 24, 33, 51, 102, 255] {
                    for beta in [0i32, 2, 4, 8, 16, 31, 60, 255] {
                        for tc0 in [0i32, 1, 3, 7, 12, 18, 25, 40] {
                            let mut a = s;
                            let mut b = s;
                            scalar_deblock_luma_segment(&mut a, bs, alpha, beta, tc0);
                            crate::simd_x86::set_force_scalar(false);
                            deblock_luma_segment(&mut b, bs, alpha, beta, tc0);
                            crate::simd_x86::set_force_scalar(true);
                            for l in 0..4 {
                                for k in 0..8 {
                                    assert_eq!(
                                        a[l][k], b[l][k],
                                        "mismatch bs={bs} alpha={alpha} beta={beta} tc0={tc0} lane={l} idx={k}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
