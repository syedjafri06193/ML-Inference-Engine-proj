//! Error codes.
//!
//! Numeric only. A `Debug` impl or a formatted message would pull
//! `core::fmt` into the binary, which is the single most common way a Rust
//! wasm module ends up ten times larger than it should be (doc 16.1). The JS
//! side maps these codes to messages.

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Error {
    Truncated = -1,
    Misaligned = -2,
    BadMagic = -3,
    BadVersion = -4,
    BadHeader = -5,
    BadTensor = -6,
    BadNode = -7,
    UnsupportedOp = -8,
    BadShape = -9,
    BadAttr = -10,
    OutOfMemory = -11,
    NotLoaded = -12,
    ScratchTooSmall = -13,
    BadDtype = -14,
}

impl Error {
    pub fn code(self) -> i32 {
        self as i32
    }
}
