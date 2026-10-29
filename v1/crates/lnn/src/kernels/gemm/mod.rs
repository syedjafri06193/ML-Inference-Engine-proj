//! Quantized GEMM (design doc section 6).
//!
//! C[m, n] = requant( bias[n] + Σ_k A[m, k] · B[k, n] )
//!
//! A is int8 activations, row-major, `lda` apart. B is int8 weights,
//! symmetric per output channel, **pre-packed by the converter** into the
//! panel layout below so no weight packing ever happens at inference time.
//! The zero-point correction `−Z_a · Σ_k B[k, n]` is folded into `bias` by
//! the converter (doc 5.2), so the hot loop is a pure int8 dot product.
//!
//! ## Packed B layout ("panel layout v1")
//!
//! N is padded to a multiple of NR = 8 and K to a multiple of 4, with zeros.
//! For each 8-column panel, for each group of 4 k: 8 columns × 4 bytes.
//!
//! ```text
//! panel nb:  [kq 0: c0k0 c0k1 c0k2 c0k3 c1k0 .. c7k3] [kq 1: ...] ...
//!            └────────── 32 bytes = two v128 ──────────┘
//! ```
//!
//! One layout serves all three kernels. It is exactly the operand shape
//! `i32x4.relaxed_dot_i8x16_i7x16_add_s` wants (four k per i32 lane), and the
//! baseline kernel reaches the k-pair shape `i32x4.dot_i16x8_s` wants with a
//! single byte shuffle. The alternative — a layout per ISA — would make the
//! model file depend on the browser that loads it.
//!
//! ## Blocking
//!
//! A is packed MC = 64 rows at a time into scratch as 4-row panels; for each
//! B panel, the 4x8 microkernel is swept down those rows while the B panel
//! (Kp × 8 bytes) stays hot in L1. There is no KC blocking: every layer in
//! the target models has K ≤ 1280, so a whole B panel is at most 10 KB.

#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
pub mod relaxed;
pub mod scalar;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod simd128;

use crate::quant::requantize;

pub const MR: usize = 4;
pub const NR: usize = 8;
pub const MC: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Scalar = 0,
    Simd128 = 1,
    Relaxed = 2,
}

/// The best kernel this build contains, before looking at the model.
pub const fn best_isa() -> Isa {
    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
    {
        return Isa::Relaxed;
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(target_feature = "relaxed-simd")
    ))]
    {
        return Isa::Simd128;
    }
    #[allow(unreachable_code)]
    Isa::Scalar
}

/// The kernel to use for a model.
///
/// The relaxed kernel is only deterministic when every weight fits in seven
/// bits (see `relaxed.rs`). A model converted without `--relaxed` has full
/// 8-bit weights, so the relaxed build quietly steps down to the baseline
/// SIMD kernel for it rather than produce numbers that differ between an
/// Intel and an ARM laptop.
pub fn isa_for(seven_bit_weights: bool) -> Isa {
    match best_isa() {
        Isa::Relaxed if !seven_bit_weights => {
            if cfg!(target_feature = "simd128") {
                Isa::Simd128
            } else {
                Isa::Scalar
            }
        }
        isa => isa,
    }
}

#[inline(always)]
pub const fn round_up(x: usize, to: usize) -> usize {
    (x + to - 1) / to * to
}

/// Bytes of packed B for an N×K weight.
pub const fn packed_b_bytes(n: usize, k: usize) -> usize {
    round_up(n, NR) * round_up(k, 4)
}

/// Scratch the GEMM itself needs (A panels plus per-row sums).
pub const fn gemm_scratch_bytes(k: usize) -> usize {
    MC * round_up(k, 4) + MC * 4
}

pub struct Gemm {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: *const i8,
    pub lda: usize,
    pub b: *const i8,
    pub bias: *const i32,
    pub mult: *const i32,
    pub shift: *const i32,
    pub out: *mut i8,
    pub ldc: usize,
    pub zero_point: i32,
    pub lo: i32,
    pub hi: i32,
}

/// A microkernel computes one MR×NR tile over the full packed depth.
/// `a`: packed A panel (kq × 16 bytes). `b`: packed B panel (kq × 32 bytes).
/// Output: `acc[r * NR + c]`, raw int32 dot products.
pub type Microkernel = unsafe fn(a: *const i8, b: *const i8, kq: usize, acc: &mut [i32; MR * NR]);

/// The single-row variant. Every FullyConnected layer at batch 1 has M = 1,
/// where the 4x8 kernel would compute three rows of padding for every real
/// one; this one holds two accumulators instead of eight and reads the same
/// packed B.
pub type Microkernel1 = unsafe fn(a: *const i8, b: *const i8, kq: usize, acc: &mut [i32; NR]);

fn microkernel1(isa: Isa) -> Microkernel1 {
    match isa {
        #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
        Isa::Relaxed => relaxed::microkernel_1xn,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::microkernel_1xn,
        _ => scalar::microkernel_1xn,
    }
}

fn microkernel(isa: Isa) -> Microkernel {
    match isa {
        #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
        Isa::Relaxed => relaxed::microkernel,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::microkernel,
        _ => scalar::microkernel,
    }
}

/// Pack `rows` rows of A (starting at `a`) into 4-row panels in `dst`.
/// Rows past `rows` and columns past `k` are zero. Returns nothing; also
/// writes per-row sums of the packed values to `sums` (the relaxed kernel's
/// correction term needs them; the others ignore them).
///
/// # Safety
/// `a` must be readable for `rows` rows of `k` bytes at stride `lda`;
/// `dst` must hold `round_up(rows, 4) * round_up(k, 4)` bytes and `sums`
/// `round_up(rows, 4)` i32.
pub unsafe fn pack_a(
    a: *const i8,
    lda: usize,
    rows: usize,
    k: usize,
    dst: *mut i8,
    sums: *mut i32,
) {
    let kp = round_up(k, 4);
    let mut mb = 0;
    while mb * MR < rows {
        let panel = dst.add(mb * MR * kp);
        for r in 0..MR {
            let row = mb * MR + r;
            let mut sum = 0i32;
            let mut kq = 0;
            while kq * 4 < kp {
                for i in 0..4 {
                    let kk = kq * 4 + i;
                    let v = if row < rows && kk < k {
                        *a.add(row * lda + kk)
                    } else {
                        0
                    };
                    *panel.add(kq * 16 + r * 4 + i) = v;
                    sum += v as i32;
                }
                kq += 1;
            }
            *sums.add(mb * MR + r) = sum;
        }
        mb += 1;
    }
}

/// Run a GEMM with requantization.
///
/// # Safety
/// All pointers must be valid for the shapes in `g`; `scratch` must hold
/// [`gemm_scratch_bytes`]`(g.k)` bytes and be 16-byte aligned. The graph
/// validator checks every one of these before a model is allowed to run.
pub unsafe fn gemm(g: &Gemm, scratch: *mut u8, isa: Isa) {
    let kp = round_up(g.k, 4);
    let kq = kp / 4;
    let np = round_up(g.n, NR);
    let kernel = microkernel(isa);
    let packed = scratch as *mut i8;
    let sums = scratch.add(MC * kp) as *mut i32;
    // The relaxed kernel feeds weights to the instruction as `w + 64`, so each
    // raw dot product carries an extra `64 · Σ_k a` to take back out.
    let correction = if isa == Isa::Relaxed { 64 } else { 0 };
    let mut acc = [0i32; MR * NR];

    if g.m == 1 {
        gemv(g, packed, sums, kq, np, kp, isa, correction);
        return;
    }

    let mut m0 = 0;
    while m0 < g.m {
        let mc = if g.m - m0 < MC { g.m - m0 } else { MC };
        pack_a(g.a.add(m0 * g.lda), g.lda, mc, g.k, packed, sums);

        let mut nb = 0;
        while nb * NR < np {
            let panel_b = g.b.add(nb * NR * kp);
            let n0 = nb * NR;
            let cols = if g.n - n0 < NR { g.n - n0 } else { NR };

            let mut mb = 0;
            while mb * MR < mc {
                kernel(packed.add(mb * MR * kp), panel_b, kq, &mut acc);

                let rows = if mc - mb * MR < MR { mc - mb * MR } else { MR };
                for r in 0..rows {
                    let row = m0 + mb * MR + r;
                    let fix = correction * *sums.add(mb * MR + r);
                    let out = g.out.add(row * g.ldc + n0);
                    for c in 0..cols {
                        let n = n0 + c;
                        let v = acc[r * NR + c] - fix + *g.bias.add(n);
                        *out.add(c) = requantize(
                            v,
                            *g.mult.add(n),
                            *g.shift.add(n),
                            g.zero_point,
                            g.lo,
                            g.hi,
                        );
                    }
                }
                mb += 1;
            }
            nb += 1;
        }
        m0 += MC;
    }
}

/// The M = 1 case: one packed row swept across the B panels.
#[allow(clippy::too_many_arguments)]
unsafe fn gemv(
    g: &Gemm,
    packed: *mut i8,
    sums: *mut i32,
    kq: usize,
    np: usize,
    kp: usize,
    isa: Isa,
    correction: i32,
) {
    let kernel = microkernel1(isa);
    // Pack the single row: copy, then zero out to a multiple of 4.
    core::ptr::copy_nonoverlapping(g.a, packed, g.k);
    for i in g.k..kp {
        *packed.add(i) = 0;
    }
    let mut sum = 0i32;
    for i in 0..g.k {
        sum += *packed.add(i) as i32;
    }
    *sums = sum;
    let fix = correction * sum;

    let mut acc = [0i32; NR];
    let mut nb = 0;
    while nb * NR < np {
        kernel(packed, g.b.add(nb * NR * kp), kq, &mut acc);
        let n0 = nb * NR;
        let cols = if g.n - n0 < NR { g.n - n0 } else { NR };
        let out = g.out.add(n0);
        for c in 0..cols {
            let n = n0 + c;
            let v = acc[c] - fix + *g.bias.add(n);
            *out.add(c) = requantize(v, *g.mult.add(n), *g.shift.add(n), g.zero_point, g.lo, g.hi);
        }
        nb += 1;
    }
}

/// Pack an N×K row-major int8 weight matrix into panel layout v1. The
/// converter does this in Python; this copy exists for the Rust tests.
#[cfg(test)]
pub fn pack_b(w: &[i8], n: usize, k: usize) -> std::vec::Vec<i8> {
    let kp = round_up(k, 4);
    let np = round_up(n, NR);
    let mut out = std::vec![0i8; np * kp];
    for nb in 0..np / NR {
        for q in 0..kp / 4 {
            for c in 0..NR {
                for i in 0..4 {
                    let (nn, kk) = (nb * NR + c, q * 4 + i);
                    if nn < n && kk < k {
                        out[nb * NR * kp + q * 32 + c * 4 + i] = w[nn * k + kk];
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::requantize;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> i8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 56) as i8
        }
    }

    fn reference(
        a: &[i8],
        w: &[i8],
        m: usize,
        n: usize,
        k: usize,
        bias: &[i32],
        mult: &[i32],
        shift: &[i32],
        zp: i32,
    ) -> std::vec::Vec<i8> {
        let mut out = std::vec![0i8; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = bias[j];
                for kk in 0..k {
                    acc += a[i * k + kk] as i32 * w[j * k + kk] as i32;
                }
                out[i * n + j] = requantize(acc, mult[j], shift[j], zp, -128, 127);
            }
        }
        out
    }

    #[test]
    fn packed_gemm_matches_the_naive_loop_on_awkward_shapes() {
        // Shapes chosen to hit every edge: M not a multiple of 4 or of MC,
        // N not a multiple of 8, K not a multiple of 4, and 1s everywhere.
        let shapes = [
            (1, 1, 1),
            (3, 5, 7),
            (4, 8, 4),
            (65, 9, 13),
            (130, 17, 33),
            (7, 64, 100),
            (1, 10, 64),
        ];
        let mut rng = Lcg(42);
        for &(m, n, k) in &shapes {
            let a: std::vec::Vec<i8> = (0..m * k).map(|_| rng.next()).collect();
            let w: std::vec::Vec<i8> = (0..n * k).map(|_| rng.next()).collect();
            let bias: std::vec::Vec<i32> = (0..n).map(|i| (i as i32 - 3) * 997).collect();
            let mult: std::vec::Vec<i32> = (0..n)
                .map(|i| 1_100_000_000 + i as i32 * 7_000_000)
                .collect();
            let shift: std::vec::Vec<i32> = (0..n).map(|i| -(6 + (i % 3) as i32)).collect();
            let zp = -7;

            let bp = pack_b(&w, n, k);
            let mut out = std::vec![0i8; m * n];
            let mut scratch = std::vec![0u8; gemm_scratch_bytes(k) + 16];
            let base = scratch.as_mut_ptr();
            let aligned = unsafe { base.add(base.align_offset(16)) };
            let g = Gemm {
                m,
                n,
                k,
                a: a.as_ptr(),
                lda: k,
                b: bp.as_ptr(),
                bias: bias.as_ptr(),
                mult: mult.as_ptr(),
                shift: shift.as_ptr(),
                out: out.as_mut_ptr(),
                ldc: n,
                zero_point: zp,
                lo: -128,
                hi: 127,
            };
            unsafe { gemm(&g, aligned, Isa::Scalar) };
            let want = reference(&a, &w, m, n, k, &bias, &mult, &shift, zp);
            for i in 0..m * n {
                assert_eq!(
                    out[i],
                    want[i],
                    "shape {:?}: first mismatch at {} ({} vs {})",
                    (m, n, k),
                    i,
                    out[i],
                    want[i]
                );
            }
        }
    }
}
