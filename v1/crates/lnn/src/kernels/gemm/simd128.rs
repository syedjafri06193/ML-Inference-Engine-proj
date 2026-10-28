//! Baseline-SIMD microkernel, 4×8 register tile (design doc 6.2–6.3).
//!
//! Baseline wasm SIMD has no int8 dot product. The closest thing is
//! `i32x4.dot_i16x8_s` (pmaddwd): 8 × i16 products summed in adjacent pairs
//! into 4 × i32. So each step works on a *pair* of k values:
//!
//! * A: one row's (a_k, a_k+1), widened to i16, splatted to all four lanes.
//! * B: for four columns, (b_k[c], b_k+1[c]) interleaved as i16.
//! * `dot` then gives, per lane c, `a_k·b_k[c] + a_k+1·b_k+1[c]`.
//!
//! The packed layout stores four k per column (for the relaxed kernel), so a
//! B load is `c0k0 c0k1 c0k2 c0k3 c1k0 …`; one byte shuffle regroups it into
//! the (k0,k1) pairs for four columns followed by the (k2,k3) pairs, and the
//! two halves widen straight into the two i16 operands.
//!
//! Accumulators: 4 rows × 2 halves of 8 columns = 8 v128, which leaves room
//! for operands on x86-64's 16 xmm registers. 8×8 would need 16 accumulators
//! and spill on every iteration (doc 6.2).

use super::{MR, NR};
use core::arch::wasm32::*;

/// # Safety
/// `a` must hold `kq * 16` bytes and `b` `kq * 32` bytes.
#[inline(never)]
pub unsafe fn microkernel(a: *const i8, b: *const i8, kq: usize, acc: &mut [i32; MR * NR]) {
    let z = i32x4_splat(0);
    let (mut c00, mut c01, mut c10, mut c11) = (z, z, z, z);
    let (mut c20, mut c21, mut c30, mut c31) = (z, z, z, z);

    let mut q = 0;
    while q < kq {
        let av = v128_load(a.add(q * 16) as *const v128);
        // rows 0,1 and rows 2,3 as i16: [r k0 k1 k2 k3] × 2
        let a01 = i16x8_extend_low_i8x16(av);
        let a23 = i16x8_extend_high_i8x16(av);

        let b0 = v128_load(b.add(q * 32) as *const v128);
        let b1 = v128_load(b.add(q * 32 + 16) as *const v128);
        // (k0,k1) pairs for 4 columns, then (k2,k3) pairs.
        let s0 = i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15>(b0, b0);
        let s1 = i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15>(b1, b1);
        let b0_01 = i16x8_extend_low_i8x16(s0);
        let b0_23 = i16x8_extend_high_i8x16(s0);
        let b1_01 = i16x8_extend_low_i8x16(s1);
        let b1_23 = i16x8_extend_high_i8x16(s1);

        // Row r's (k0,k1) is i32 lane 2r of a01/a23; (k2,k3) is lane 2r+1.
        let r0_01 = i32x4_shuffle::<0, 0, 0, 0>(a01, a01);
        let r0_23 = i32x4_shuffle::<1, 1, 1, 1>(a01, a01);
        let r1_01 = i32x4_shuffle::<2, 2, 2, 2>(a01, a01);
        let r1_23 = i32x4_shuffle::<3, 3, 3, 3>(a01, a01);
        let r2_01 = i32x4_shuffle::<0, 0, 0, 0>(a23, a23);
        let r2_23 = i32x4_shuffle::<1, 1, 1, 1>(a23, a23);
        let r3_01 = i32x4_shuffle::<2, 2, 2, 2>(a23, a23);
        let r3_23 = i32x4_shuffle::<3, 3, 3, 3>(a23, a23);

        c00 = i32x4_add(
            c00,
            i32x4_add(i32x4_dot_i16x8(r0_01, b0_01), i32x4_dot_i16x8(r0_23, b0_23)),
        );
        c01 = i32x4_add(
            c01,
            i32x4_add(i32x4_dot_i16x8(r0_01, b1_01), i32x4_dot_i16x8(r0_23, b1_23)),
        );
        c10 = i32x4_add(
            c10,
            i32x4_add(i32x4_dot_i16x8(r1_01, b0_01), i32x4_dot_i16x8(r1_23, b0_23)),
        );
        c11 = i32x4_add(
            c11,
            i32x4_add(i32x4_dot_i16x8(r1_01, b1_01), i32x4_dot_i16x8(r1_23, b1_23)),
        );
        c20 = i32x4_add(
            c20,
            i32x4_add(i32x4_dot_i16x8(r2_01, b0_01), i32x4_dot_i16x8(r2_23, b0_23)),
        );
        c21 = i32x4_add(
            c21,
            i32x4_add(i32x4_dot_i16x8(r2_01, b1_01), i32x4_dot_i16x8(r2_23, b1_23)),
        );
        c30 = i32x4_add(
            c30,
            i32x4_add(i32x4_dot_i16x8(r3_01, b0_01), i32x4_dot_i16x8(r3_23, b0_23)),
        );
        c31 = i32x4_add(
            c31,
            i32x4_add(i32x4_dot_i16x8(r3_01, b1_01), i32x4_dot_i16x8(r3_23, b1_23)),
        );
        q += 1;
    }

    let out = acc.as_mut_ptr() as *mut v128;
    v128_store(out, c00);
    v128_store(out.add(1), c01);
    v128_store(out.add(2), c10);
    v128_store(out.add(3), c11);
    v128_store(out.add(4), c20);
    v128_store(out.add(5), c21);
    v128_store(out.add(6), c30);
    v128_store(out.add(7), c31);
}

/// The 4x8 kernel's single-row case: two accumulators instead of eight.
///
/// # Safety
/// `a` holds `kq * 4` bytes and `b` `kq * 32`.
#[inline(never)]
pub unsafe fn microkernel_1xn(a: *const i8, b: *const i8, kq: usize, acc: &mut [i32; NR]) {
    let mut c0 = i32x4_splat(0);
    let mut c1 = i32x4_splat(0);
    let mut q = 0;
    while q < kq {
        // The row's four k values, widened to i16 and splatted as (k0,k1)
        // and (k2,k3) pairs.
        let av = i16x8_extend_low_i8x16(v128_load32_zero(a.add(q * 4) as *const u32));
        let a01 = i32x4_shuffle::<0, 0, 0, 0>(av, av);
        let a23 = i32x4_shuffle::<1, 1, 1, 1>(av, av);

        let b0 = v128_load(b.add(q * 32) as *const v128);
        let b1 = v128_load(b.add(q * 32 + 16) as *const v128);
        let s0 = i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15>(b0, b0);
        let s1 = i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15>(b1, b1);

        c0 = i32x4_add(
            c0,
            i32x4_add(
                i32x4_dot_i16x8(a01, i16x8_extend_low_i8x16(s0)),
                i32x4_dot_i16x8(a23, i16x8_extend_high_i8x16(s0)),
            ),
        );
        c1 = i32x4_add(
            c1,
            i32x4_add(
                i32x4_dot_i16x8(a01, i16x8_extend_low_i8x16(s1)),
                i32x4_dot_i16x8(a23, i16x8_extend_high_i8x16(s1)),
            ),
        );
        q += 1;
    }
    let out = acc.as_mut_ptr() as *mut v128;
    v128_store(out, c0);
    v128_store(out.add(1), c1);
}
