//! `lnn` — a dependency-free int8 inference runtime for wasm32.
//!
//! The whole public interface is the nine `extern "C"` functions at the
//! bottom of this file. No `wasm-bindgen`, no generated glue: the JS side
//! writes bytes into linear memory at a pointer this module hands out, calls
//! `lnn_run`, and reads the result back (design doc 13.1).

#![cfg_attr(not(test), no_std)]
#![allow(clippy::missing_safety_doc)]

#[cfg(feature = "bench")]
pub mod bench;
pub mod error;
pub mod graph;
pub mod kernels;
pub mod model;
pub mod quant;

#[cfg(all(target_arch = "wasm32", not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // No message, no formatting: `core::fmt` is the single largest thing that
    // sneaks into a Rust wasm binary (doc 16.1). Everything reachable from
    // the ABI returns an error code instead of panicking, so arriving here
    // means a bug in this crate, and a trap is the honest signal.
    core::arch::wasm32::unreachable()
}

use error::Error;
use graph::Buffers;
use kernels::gemm::Isa;
use model::*;

/// Everything the runtime holds between calls.
struct Engine {
    model_ptr: *const u8,
    model_len: usize,
    arena: *mut u8,
    scratch: *mut u8,
    isa: Isa,
}

static mut ENGINE: Option<Engine> = None;

#[allow(static_mut_refs)]
fn engine() -> Option<&'static Engine> {
    unsafe { ENGINE.as_ref() }
}

// ---------------------------------------------------------------- allocator

/// A bump allocator over the wasm heap.
///
/// The runtime allocates exactly three times — model, arena, scratch — all
/// at load time, and never during `run()` (doc 4.2, 8.2). A general-purpose
/// allocator would be several KB of code to support a call pattern this
/// simple, so there isn't one.
#[cfg(target_arch = "wasm32")]
mod bump {
    extern "C" {
        static __heap_base: u8;
    }

    const PAGE: usize = 65536;
    static mut NEXT: usize = 0;

    pub fn alloc(n: usize) -> *mut u8 {
        unsafe {
            if NEXT == 0 {
                NEXT = (&__heap_base as *const u8 as usize + 15) & !15;
            }
            let start = (NEXT + 15) & !15;
            let end = match start.checked_add(n) {
                Some(e) => e,
                None => return core::ptr::null_mut(),
            };
            let have = core::arch::wasm32::memory_size(0) * PAGE;
            if end > have {
                let need_pages = (end - have + PAGE - 1) / PAGE;
                // grow returns usize::MAX on failure; the caller turns that
                // into an error code rather than letting a later store trap.
                if core::arch::wasm32::memory_grow(0, need_pages) == usize::MAX {
                    return core::ptr::null_mut();
                }
            }
            NEXT = end;
            start as *mut u8
        }
    }

    /// Bytes handed out so far — reported as peak memory by the JS API.
    pub fn used() -> usize {
        unsafe { NEXT }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod bump {
    // Host builds (cargo test) use the system allocator through a leaked Vec
    // so the same ABI code paths can be exercised natively.
    extern crate std;
    use std::vec;
    static mut USED: usize = 0;

    pub fn alloc(n: usize) -> *mut u8 {
        unsafe {
            USED += n;
            let mut v = vec![0u8; n + 16];
            let p = v.as_mut_ptr();
            let off = p.align_offset(16);
            std::mem::forget(v);
            p.add(off)
        }
    }

    pub fn used() -> usize {
        unsafe { USED }
    }
}

// ----------------------------------------------------------------- loading

/// Parse, validate and prepare a model that already sits in linear memory.
fn load_impl(ptr: *const u8, len: usize) -> Result<(), Error> {
    let buf = unsafe { core::slice::from_raw_parts(ptr, len) };
    let m = Model::parse(buf)?;
    graph::validate(&m)?;

    let arena = bump::alloc(m.header.arena_bytes as usize);
    if arena.is_null() {
        return Err(Error::OutOfMemory);
    }
    // The scratch buffer is read by SIMD loads, so it has to be 16-byte
    // aligned; the bump allocator always is, but say so where it matters.
    let scratch_bytes = m.header.scratch_bytes.max(64) as usize;
    let scratch = bump::alloc(scratch_bytes);
    if scratch.is_null() {
        return Err(Error::OutOfMemory);
    }
    unsafe {
        core::ptr::write_bytes(arena, 0, m.header.arena_bytes as usize);
        ENGINE = Some(Engine {
            model_ptr: ptr,
            model_len: len,
            arena,
            scratch,
            isa: kernels::gemm::isa_for(m.flags() & FLAG_WEIGHTS_7BIT != 0),
        });
    }
    Ok(())
}

fn with_model<T>(f: impl FnOnce(&Model, &Engine) -> T) -> Option<T> {
    let e = engine()?;
    let buf = unsafe { core::slice::from_raw_parts(e.model_ptr, e.model_len) };
    // Already validated at load time; re-parsing is a few field reads.
    let m = Model::parse(buf).ok()?;
    Some(f(&m, e))
}

/// Byte offset and length of an input or output tensor within the arena.
fn io_span(kind: u8, index: u32) -> Option<(usize, usize)> {
    with_model(|m, _| {
        let t = m.io_tensor(kind, index)?;
        let t = m.tensor(t);
        Some((t.offset as usize, t.numel() * t.elem_bytes()))
    })
    .flatten()
}

// --------------------------------------------------------------- public ABI

/// Reserve `n` bytes of linear memory. Returns null if memory cannot grow.
#[no_mangle]
pub extern "C" fn lnn_alloc(n: usize) -> *mut u8 {
    bump::alloc(n)
}

/// Load a `.lnn` model already written at `ptr`. 0 on success, negative
/// [`Error`] code otherwise. The buffer must stay alive and unmodified.
#[no_mangle]
pub unsafe extern "C" fn lnn_load(ptr: *const u8, len: usize) -> i32 {
    match load_impl(ptr, len) {
        Ok(()) => 0,
        Err(e) => {
            ENGINE = None;
            e.code()
        }
    }
}

/// Pointer to input `index`'s buffer, for JS to write fp32 samples into.
#[no_mangle]
pub extern "C" fn lnn_input_ptr(index: u32) -> *mut u8 {
    match (engine(), io_span(KIND_INPUT, index)) {
        (Some(e), Some((off, _))) => unsafe { e.arena.add(off) },
        _ => core::ptr::null_mut(),
    }
}

/// Input `index`'s size in bytes.
#[no_mangle]
pub extern "C" fn lnn_input_len(index: u32) -> u32 {
    io_span(KIND_INPUT, index)
        .map(|(_, n)| n as u32)
        .unwrap_or(0)
}

/// Execute the graph. 0 on success, negative on error.
#[no_mangle]
pub extern "C" fn lnn_run() -> i32 {
    let ok = with_model(|m, e| {
        let b = Buffers {
            arena: e.arena,
            scratch: e.scratch,
            isa: e.isa,
        };
        unsafe { graph::run(m, &b) };
    });
    match ok {
        Some(()) => 0,
        None => Error::NotLoaded.code(),
    }
}

#[no_mangle]
pub extern "C" fn lnn_output_ptr(index: u32) -> *const u8 {
    match (engine(), io_span(KIND_OUTPUT, index)) {
        (Some(e), Some((off, _))) => unsafe { e.arena.add(off) },
        _ => core::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn lnn_output_len(index: u32) -> u32 {
    io_span(KIND_OUTPUT, index)
        .map(|(_, n)| n as u32)
        .unwrap_or(0)
}

/// Which kernel the loaded model actually runs on: 0 scalar, 1 simd128,
/// 2 relaxed. Not which build this is — a relaxed build steps down to
/// simd128 for a model whose weights are not 7-bit (see `gemm/relaxed.rs`),
/// and "SIMD is no faster" is usually this (doc 17.5).
#[no_mangle]
pub extern "C" fn lnn_isa() -> u32 {
    engine().map(|e| e.isa as u32).unwrap_or(0)
}

/// Pointer to any tensor's buffer, by index.
///
/// This is the layer-bisection hook from doc 17.1: dump intermediate
/// activations and compare them to the reference to find the *first* node
/// that diverges. `tests/test_conformance.py` uses it, and it costs about
/// twenty bytes of binary.
#[no_mangle]
pub extern "C" fn lnn_tensor_ptr(index: u32) -> *const u8 {
    with_model(|m, e| {
        if index >= m.header.n_tensors {
            return core::ptr::null();
        }
        let t = m.tensor(index);
        if t.is_activation() {
            unsafe { e.arena.add(t.offset as usize) }
        } else {
            unsafe { m.weights().as_ptr().add(t.offset as usize) }
        }
    })
    .unwrap_or(core::ptr::null())
}

/// Bytes of linear memory handed out so far.
#[no_mangle]
pub extern "C" fn lnn_memory_used() -> u32 {
    bump::used() as u32
}
