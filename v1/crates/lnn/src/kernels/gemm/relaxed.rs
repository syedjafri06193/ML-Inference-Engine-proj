//! Relaxed-SIMD microkernel (design doc 6.4).
//!
//! `i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, c)`: per i32 lane, four int8
//! products summed and added to `c` — sixteen multiply-accumulates in one
//! instruction, mapping to VPDPBUSD / pmaddubsw on x86 and SDOT on ARM.
//!
//! ## The determinism rule — and a correction to the design doc
//!
//! The instruction's result is only defined when **the top bit of every byte
//! of `b` is clear**: x86 treats that operand as unsigned, ARM as signed, and
//! they agree exactly on 0..=127. The design doc says to restrict weights to
//! −64..63, but −64..−1 all have the top bit set (−64 is 0xC0), so that range
//! gives different answers on Intel and on an M-series Mac.
//!
//! What this kernel does instead: the converter's `--relaxed` mode limits
//! weights to −63..63, and the kernel adds 64 to each weight byte before the
//! instruction, so the operand it sees is 1..127 — top bit always clear. The
//! extra `64 · Σ_k a[k]` that puts into every accumulator is removed in the
//! GEMM epilogue from the per-row sums computed during A packing. The pair
//! sums inside the instruction are at most 2 · 128 · 127 = 32512, so the
//! i16 intermediate x86 uses cannot saturate either.

use super::{MR, NR};
use core::arch::wasm32::*;

/// # Safety
/// `a` must hold `kq * 16` bytes and `b` `kq * 32` bytes; every weight byte
/// in `b` must lie in −64..=63 (the loader enforces this by only selecting
/// this kernel for models flagged as 7-bit).
#[inline(never)]
pub unsafe fn microkernel(a: *const i8, b: *const i8, kq: usize, acc: &mut [i32; MR * NR]) {
    let z = i32x4_splat(0);
    let bias = i8x16_splat(64);
    let (mut c00, mut c01, mut c10, mut c11) = (z, z, z, z);
    let (mut c20, mut c21, mut c30, mut c31) = (z, z, z, z);

    let mut q = 0;
    while q < kq {
        let av = v128_load(a.add(q * 16) as *const v128);
        let b0 = i8x16_add(v128_load(b.add(q * 32) as *const v128), bias);
        let b1 = i8x16_add(v128_load(b.add(q * 32 + 16) as *const v128), bias);

        // Row r's four k values, replicated into every i32 lane.
        let r0 = i32x4_shuffle::<0, 0, 0, 0>(av, av);
        let r1 = i32x4_shuffle::<1, 1, 1, 1>(av, av);
        let r2 = i32x4_shuffle::<2, 2, 2, 2>(av, av);
        let r3 = i32x4_shuffle::<3, 3, 3, 3>(av, av);

        c00 = i32x4_relaxed_dot_i8x16_i7x16_add(r0, b0, c00);
        c01 = i32x4_relaxed_dot_i8x16_i7x16_add(r0, b1, c01);
        c10 = i32x4_relaxed_dot_i8x16_i7x16_add(r1, b0, c10);
        c11 = i32x4_relaxed_dot_i8x16_i7x16_add(r1, b1, c11);
        c20 = i32x4_relaxed_dot_i8x16_i7x16_add(r2, b0, c20);
        c21 = i32x4_relaxed_dot_i8x16_i7x16_add(r2, b1, c21);
        c30 = i32x4_relaxed_dot_i8x16_i7x16_add(r3, b0, c30);
        c31 = i32x4_relaxed_dot_i8x16_i7x16_add(r3, b1, c31);
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

/// Single-row relaxed kernel; the +64 bias and its correction work exactly
/// as in the 4x8 case.
///
/// # Safety
/// `a` holds `kq * 4` bytes and `b` `kq * 32`.
#[inline(never)]
pub unsafe fn microkernel_1xn(a: *const i8, b: *const i8, kq: usize, acc: &mut [i32; NR]) {
    let bias = i8x16_splat(64);
    let mut c0 = i32x4_splat(0);
    let mut c1 = i32x4_splat(0);
    let mut q = 0;
    while q < kq {
        let av = v128_load32_zero(a.add(q * 4) as *const u32);
        let r0 = i32x4_shuffle::<0, 0, 0, 0>(av, av);
        let b0 = i8x16_add(v128_load(b.add(q * 32) as *const v128), bias);
        let b1 = i8x16_add(v128_load(b.add(q * 32 + 16) as *const v128), bias);
        c0 = i32x4_relaxed_dot_i8x16_i7x16_add(r0, b0, c0);
        c1 = i32x4_relaxed_dot_i8x16_i7x16_add(r0, b1, c1);
        q += 1;
    }
    let out = acc.as_mut_ptr() as *mut v128;
    v128_store(out, c0);
    v128_store(out.add(1), c1);
}
