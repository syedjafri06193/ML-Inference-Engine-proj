//! Benchmark-only exports (built with `--features bench`).
//!
//! These exist to measure the optimization ladder from design doc M4 — naive
//! loops, then blocking and packing, then the SIMD microkernels — on the same
//! matrix shapes, in the same engine, so the numbers in `1—Documentation/v1/benchmarks.md`
//! are comparable. They are not part of the shipped runtime: the ladder's
//! lower rungs are code nobody should run in production.

use crate::kernels::gemm::{self, Gemm, Isa};
use crate::quant::requantize;

const MAX_BYTES: usize = 16 << 20;

static mut A: *mut i8 = core::ptr::null_mut();
static mut B: *mut i8 = core::ptr::null_mut();
static mut BP: *mut i8 = core::ptr::null_mut();
static mut C: *mut i8 = core::ptr::null_mut();
static mut BIAS: *mut i32 = core::ptr::null_mut();
static mut MULT: *mut i32 = core::ptr::null_mut();
static mut SHIFT: *mut i32 = core::ptr::null_mut();
static mut SCRATCH: *mut u8 = core::ptr::null_mut();
static mut DIMS: (usize, usize, usize) = (0, 0, 0);

/// Allocate and fill buffers for an m×n×k GEMM. Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn lnn_bench_setup(m: usize, n: usize, k: usize) -> i32 {
    if m * k > MAX_BYTES || n * k > MAX_BYTES || m * n > MAX_BYTES {
        return -1;
    }
    A = crate::lnn_alloc(m * k) as *mut i8;
    B = crate::lnn_alloc(n * k) as *mut i8;
    BP = crate::lnn_alloc(gemm::packed_b_bytes(n, k)) as *mut i8;
    C = crate::lnn_alloc(m * n) as *mut i8;
    BIAS = crate::lnn_alloc(4 * gemm::round_up(n, 8)) as *mut i32;
    MULT = crate::lnn_alloc(4 * gemm::round_up(n, 8)) as *mut i32;
    SHIFT = crate::lnn_alloc(4 * gemm::round_up(n, 8)) as *mut i32;
    SCRATCH = crate::lnn_alloc(gemm::gemm_scratch_bytes(k));
    if A.is_null() || B.is_null() || BP.is_null() || C.is_null() || SCRATCH.is_null() {
        return -2;
    }

    // A cheap deterministic fill; the values only have to be representative,
    // not random.
    let mut s: u32 = 12345;
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s >> 24) as i32 - 128) as i8
    };
    for i in 0..m * k {
        *A.add(i) = next();
    }
    for i in 0..n * k {
        // 7-bit, so the same weights are valid for the relaxed kernel.
        *B.add(i) = (next() as i32 / 2) as i8;
    }
    for i in 0..gemm::round_up(n, 8) {
        *BIAS.add(i) = 0;
        *MULT.add(i) = 1 << 30;
        *SHIFT.add(i) = -8;
    }
    pack_b(n, k);
    DIMS = (m, n, k);
    0
}

unsafe fn pack_b(n: usize, k: usize) {
    let kp = gemm::round_up(k, 4);
    let np = gemm::round_up(n, gemm::NR);
    for nb in 0..np / gemm::NR {
        for q in 0..kp / 4 {
            for c in 0..gemm::NR {
                for i in 0..4 {
                    let (nn, kk) = (nb * gemm::NR + c, q * 4 + i);
                    let v = if nn < n && kk < k {
                        *B.add(nn * k + kk)
                    } else {
                        0
                    };
                    *BP.add(nb * gemm::NR * kp + q * 32 + c * 4 + i) = v;
                }
            }
        }
    }
}

/// Run one GEMM with the given rung of the ladder:
/// 0 = naive triple loop, no packing or blocking
/// 1 = packed + blocked, scalar microkernel
/// 2 = packed + blocked, simd128 microkernel
/// 3 = packed + blocked, relaxed-simd microkernel
#[no_mangle]
pub unsafe extern "C" fn lnn_bench_gemm(rung: u32) -> i32 {
    let (m, n, k) = DIMS;
    if m == 0 {
        return -1;
    }
    match rung {
        0 => naive(m, n, k),
        1 => run(Isa::Scalar, m, n, k),
        2 => {
            if cfg!(target_feature = "simd128") {
                run(Isa::Simd128, m, n, k)
            } else {
                return -3;
            }
        }
        3 => {
            if cfg!(target_feature = "relaxed-simd") {
                run(Isa::Relaxed, m, n, k)
            } else {
                return -3;
            }
        }
        _ => return -2,
    }
    0
}

unsafe fn run(isa: Isa, m: usize, n: usize, k: usize) {
    let g = Gemm {
        m,
        n,
        k,
        a: A,
        lda: k,
        b: BP,
        bias: BIAS,
        mult: MULT,
        shift: SHIFT,
        out: C,
        ldc: n,
        zero_point: 0,
        lo: -128,
        hi: 127,
    };
    gemm::gemm(&g, SCRATCH, isa);
}

/// The starting point from M4: no packing, no blocking, no vectorization.
unsafe fn naive(m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = *BIAS.add(j);
            for kk in 0..k {
                acc += (*A.add(i * k + kk) as i32) * (*B.add(j * k + kk) as i32);
            }
            *C.add(i * n + j) = requantize(acc, *MULT.add(j), *SHIFT.add(j), 0, -128, 127);
        }
    }
}

/// Checksum of the output, so a rung that computes the wrong thing cannot
/// look fast. Every rung must return the same value.
#[no_mangle]
pub unsafe extern "C" fn lnn_bench_checksum() -> i32 {
    let (m, n, _) = DIMS;
    let mut sum: i32 = 0;
    for i in 0..m * n {
        sum = sum.wrapping_mul(31).wrapping_add(*C.add(i) as i32);
    }
    sum
}
