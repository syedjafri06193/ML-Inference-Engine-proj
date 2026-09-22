//! The reference microkernel: plain loops over the packed layout.
//!
//! This is what the scalar build ships, and it is the kernel the SIMD kernels
//! are tested against. Keep it obviously correct rather than fast.

use super::{MR, NR};

/// # Safety
/// `a` must hold `kq * 16` bytes and `b` `kq * 32` bytes.
pub unsafe fn microkernel(a: *const i8, b: *const i8, kq: usize, acc: &mut [i32; MR * NR]) {
    *acc = [0; MR * NR];
    for q in 0..kq {
        let ap = a.add(q * 16);
        let bp = b.add(q * 32);
        for r in 0..MR {
            for c in 0..NR {
                let mut s = 0i32;
                for i in 0..4 {
                    s += (*ap.add(r * 4 + i) as i32) * (*bp.add(c * 4 + i) as i32);
                }
                acc[r * NR + c] += s;
            }
        }
    }
}

/// One row against an NR-column panel. `a` is the row, zero-padded to a
/// multiple of 4; `b` is the same packed panel the 4x8 kernel reads.
///
/// # Safety
/// `a` holds `kq * 4` bytes and `b` `kq * 32`.
pub unsafe fn microkernel_1xn(a: *const i8, b: *const i8, kq: usize, acc: &mut [i32; NR]) {
    *acc = [0; NR];
    for q in 0..kq {
        let ap = a.add(q * 4);
        let bp = b.add(q * 32);
        for c in 0..NR {
            let mut s = 0i32;
            for i in 0..4 {
                s += (*ap.add(i) as i32) * (*bp.add(c * 4 + i) as i32);
            }
            acc[c] += s;
        }
    }
}
