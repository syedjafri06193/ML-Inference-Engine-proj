//! Zero-copy `.lnn` parser (design doc sections 7 and 13.2).
//!
//! The model file is untrusted input. Every offset and length is validated
//! against the buffer before anything is dereferenced, and validation happens
//! once, at load time, so the interpreter can then run without re-checking.
//! A malformed file produces an [`Error`], never a trap or an out-of-bounds
//! read — `tests/test_fuzz.py` throws mutated files at this to prove it.

use crate::error::Error;

pub const MAGIC: u32 = 0x314E_4E4C; // "LNN1" little-endian
pub const VERSION: u32 = 1;

pub const HEADER_BYTES: usize = 64;
/// 40, not the 32 the design doc states: its field list (4 bytes of flags,
/// 16 of dims, then offset, scale_off, zero_point and name_off) already adds
/// up to 36. `nbytes` is added so the parser never has to trust arithmetic on
/// dims to know how large a packed, padded weight is.
pub const TENSOR_BYTES: usize = 40;
pub const NODE_HEADER_BYTES: usize = 8;

pub const FLAG_PREPACKED: u32 = 1 << 0;
pub const FLAG_WEIGHTS_7BIT: u32 = 1 << 1;

pub const DTYPE_I8: u8 = 0;
pub const DTYPE_U8: u8 = 1;
pub const DTYPE_I32: u8 = 2;
pub const DTYPE_F32: u8 = 3;

pub const KIND_WEIGHT: u8 = 0;
pub const KIND_ACTIVATION: u8 = 1;
pub const KIND_INPUT: u8 = 2;
pub const KIND_OUTPUT: u8 = 3;

pub const MAX_NODES: u32 = 1 << 16;
pub const MAX_TENSORS: u32 = 1 << 16;
pub const MAX_IO: usize = 8;

#[inline(always)]
fn rd_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline(always)]
fn rd_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

#[derive(Clone, Copy)]
pub struct Header {
    pub version: u32,
    pub flags: u32,
    pub n_tensors: u32,
    pub n_nodes: u32,
    pub n_inputs: u32,
    pub n_outputs: u32,
    pub arena_bytes: u32,
    pub scratch_bytes: u32,
    pub tensors_off: u32,
    pub nodes_off: u32,
    pub weights_off: u32,
    pub strings_off: u32,
    pub nodes_bytes: u32,
}

/// A decoded tensor descriptor. Reading fields byte-wise rather than casting
/// to a `#[repr(C)]` struct keeps the parser correct regardless of the
/// descriptor table's alignment and endianness assumptions.
#[derive(Clone, Copy)]
pub struct Tensor {
    pub dtype: u8,
    pub rank: u8,
    pub kind: u8,
    pub n_scales: u8,
    pub dims: [u32; 4],
    pub offset: u32,
    pub scale_off: u32,
    pub zero_point: i32,
    pub nbytes: u32,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        let mut n = 1usize;
        for i in 0..self.rank as usize {
            n = n.saturating_mul(self.dims[i] as usize);
        }
        n
    }

    pub fn elem_bytes(&self) -> usize {
        match self.dtype {
            DTYPE_I32 | DTYPE_F32 => 4,
            _ => 1,
        }
    }

    pub fn is_activation(&self) -> bool {
        self.kind != KIND_WEIGHT
    }

    /// Last dimension — channels, in NHWC.
    pub fn last_dim(&self) -> usize {
        if self.rank == 0 {
            1
        } else {
            self.dims[self.rank as usize - 1] as usize
        }
    }

    /// NHWC view: (H, W, C). Rank-2 tensors are treated as 1x1 spatial.
    pub fn hwc(&self) -> (usize, usize, usize) {
        match self.rank {
            4 => (
                self.dims[1] as usize,
                self.dims[2] as usize,
                self.dims[3] as usize,
            ),
            3 => (
                self.dims[0] as usize,
                self.dims[1] as usize,
                self.dims[2] as usize,
            ),
            2 => (1, 1, self.dims[1] as usize),
            1 => (1, 1, self.dims[0] as usize),
            _ => (1, 1, 1),
        }
    }
}

/// One node, borrowed from the file.
#[derive(Clone, Copy)]
pub struct Node<'a> {
    pub opcode: u16,
    raw_inputs: &'a [u8],
    raw_outputs: &'a [u8],
    raw_attrs: &'a [u8],
}

impl<'a> Node<'a> {
    pub fn n_in(&self) -> usize {
        self.raw_inputs.len() / 4
    }
    pub fn n_out(&self) -> usize {
        self.raw_outputs.len() / 4
    }
    pub fn input(&self, i: usize) -> u32 {
        rd_u32(self.raw_inputs, i * 4)
    }
    pub fn output(&self, i: usize) -> u32 {
        rd_u32(self.raw_outputs, i * 4)
    }
    pub fn n_attrs(&self) -> usize {
        self.raw_attrs.len() / 4
    }
    /// Attributes are little-endian i32 words.
    pub fn attr(&self, i: usize) -> i32 {
        if i >= self.n_attrs() {
            return 0;
        }
        rd_u32(self.raw_attrs, i * 4) as i32
    }
}

pub struct Model<'a> {
    pub header: Header,
    pub raw: &'a [u8],
}

impl<'a> Model<'a> {
    /// Parse and structurally validate a model.
    ///
    /// Structural here means: the header is sane, every table lies inside the
    /// buffer, every tensor's bytes lie inside the weight blob or the arena,
    /// and every node's tensor references are in range. Operator-specific
    /// validation (shapes, dtypes, attributes) is done by
    /// [`crate::graph::validate`], which the loader always calls next.
    pub fn parse(buf: &'a [u8]) -> Result<Self, Error> {
        if buf.len() < HEADER_BYTES {
            return Err(Error::Truncated);
        }
        if buf.as_ptr() as usize % 16 != 0 {
            return Err(Error::Misaligned);
        }
        if rd_u32(buf, 0) != MAGIC {
            return Err(Error::BadMagic);
        }
        let h = Header {
            version: rd_u32(buf, 4),
            flags: rd_u32(buf, 8),
            n_tensors: rd_u32(buf, 12),
            n_nodes: rd_u32(buf, 16),
            n_inputs: rd_u32(buf, 20),
            n_outputs: rd_u32(buf, 24),
            arena_bytes: rd_u32(buf, 28),
            scratch_bytes: rd_u32(buf, 32),
            tensors_off: rd_u32(buf, 36),
            nodes_off: rd_u32(buf, 40),
            weights_off: rd_u32(buf, 44),
            strings_off: rd_u32(buf, 48),
            nodes_bytes: rd_u32(buf, 52),
        };
        if h.version != VERSION {
            return Err(Error::BadVersion);
        }
        if h.n_tensors == 0 || h.n_tensors > MAX_TENSORS || h.n_nodes > MAX_NODES {
            return Err(Error::BadHeader);
        }
        if h.n_inputs == 0
            || h.n_outputs == 0
            || h.n_inputs as usize > MAX_IO
            || h.n_outputs as usize > MAX_IO
        {
            return Err(Error::BadHeader);
        }

        let len = buf.len() as u64;
        let tensors_end = h.tensors_off as u64 + h.n_tensors as u64 * TENSOR_BYTES as u64;
        if (h.tensors_off as usize) < HEADER_BYTES || tensors_end > len {
            return Err(Error::Truncated);
        }
        let nodes_end = h.nodes_off as u64 + h.nodes_bytes as u64;
        if (h.nodes_off as usize) < HEADER_BYTES || nodes_end > len || h.nodes_off % 4 != 0 {
            return Err(Error::Truncated);
        }
        if h.weights_off as u64 > len || h.weights_off % 16 != 0 {
            return Err(Error::Truncated);
        }
        if h.strings_off as u64 > len {
            return Err(Error::Truncated);
        }

        let m = Model {
            header: h,
            raw: buf,
        };
        m.validate_tensors()?;
        m.validate_nodes()?;
        Ok(m)
    }

    pub fn flags(&self) -> u32 {
        self.header.flags
    }

    pub fn weights(&self) -> &'a [u8] {
        &self.raw[self.header.weights_off as usize..]
    }

    /// Descriptor `i`. Callers only pass indices that `parse` has validated;
    /// an out-of-range index still cannot read out of bounds (slice indexing
    /// panics, which in the shipped build is an abort, not a wild read).
    pub fn tensor(&self, i: u32) -> Tensor {
        let o = self.header.tensors_off as usize + i as usize * TENSOR_BYTES;
        let b = self.raw;
        Tensor {
            dtype: b[o],
            rank: b[o + 1],
            kind: b[o + 2],
            n_scales: b[o + 3],
            dims: [
                rd_u32(b, o + 4),
                rd_u32(b, o + 8),
                rd_u32(b, o + 12),
                rd_u32(b, o + 16),
            ],
            offset: rd_u32(b, o + 20),
            scale_off: rd_u32(b, o + 24),
            zero_point: rd_u32(b, o + 28) as i32,
            nbytes: rd_u32(b, o + 36),
        }
    }

    /// First scale of a tensor (per-tensor scale, or channel 0).
    pub fn scale(&self, t: &Tensor) -> f32 {
        let w = self.weights();
        let o = t.scale_off as usize;
        f32::from_le_bytes([w[o], w[o + 1], w[o + 2], w[o + 3]])
    }

    fn validate_tensors(&self) -> Result<(), Error> {
        let wlen = self.weights().len() as u64;
        let (mut n_in, mut n_out) = (0u32, 0u32);
        for i in 0..self.header.n_tensors {
            let t = self.tensor(i);
            if t.dtype > DTYPE_F32 || t.rank == 0 || t.rank > 4 || t.kind > KIND_OUTPUT {
                return Err(Error::BadTensor);
            }
            for d in 0..t.rank as usize {
                if t.dims[d] == 0 {
                    return Err(Error::BadTensor);
                }
            }
            // numel * elem_bytes must fit in nbytes. u64 arithmetic: a
            // hostile file can choose dims whose product overflows usize.
            let mut numel = 1u64;
            for d in 0..t.rank as usize {
                numel = numel.saturating_mul(t.dims[d] as u64);
            }
            if numel.saturating_mul(t.elem_bytes() as u64) > t.nbytes as u64 {
                return Err(Error::BadTensor);
            }
            let end = t.offset as u64 + t.nbytes as u64;
            if t.is_activation() {
                if end > self.header.arena_bytes as u64 || t.offset % 16 != 0 {
                    return Err(Error::BadTensor);
                }
            } else if end > wlen || t.offset % 16 != 0 {
                return Err(Error::BadTensor);
            }
            if t.n_scales > 0 {
                let s_end = t.scale_off as u64 + 4 * t.n_scales as u64;
                if s_end > wlen || t.scale_off % 4 != 0 {
                    return Err(Error::BadTensor);
                }
                // A zero, negative, infinite or NaN scale is a division by
                // zero or a NaN waiting to happen downstream.
                let s = self.scale(&t);
                if !(s > 0.0 && s < f32::INFINITY) {
                    return Err(Error::BadTensor);
                }
            }
            match t.kind {
                KIND_INPUT => {
                    if i != n_in {
                        return Err(Error::BadTensor); // inputs come first, in order
                    }
                    n_in += 1;
                }
                KIND_OUTPUT => n_out += 1,
                _ => {}
            }
        }
        if n_in != self.header.n_inputs || n_out != self.header.n_outputs {
            return Err(Error::BadHeader);
        }
        Ok(())
    }

    fn validate_nodes(&self) -> Result<(), Error> {
        let mut c = NodeCursor::new(self);
        let mut count = 0u32;
        loop {
            match c.next_checked()? {
                Some(node) => {
                    for i in 0..node.n_in() {
                        if node.input(i) >= self.header.n_tensors {
                            return Err(Error::BadNode);
                        }
                    }
                    for i in 0..node.n_out() {
                        let o = node.output(i);
                        if o >= self.header.n_tensors || !self.tensor(o).is_activation() {
                            return Err(Error::BadNode);
                        }
                    }
                    count += 1;
                }
                None => break,
            }
        }
        if count != self.header.n_nodes {
            return Err(Error::BadNode);
        }
        Ok(())
    }

    /// Index of the `i`-th tensor of `kind` (inputs or outputs), in order.
    pub fn io_tensor(&self, kind: u8, i: u32) -> Option<u32> {
        let mut seen = 0;
        for t in 0..self.header.n_tensors {
            if self.tensor(t).kind == kind {
                if seen == i {
                    return Some(t);
                }
                seen += 1;
            }
        }
        None
    }
}

/// Walks the node table. `next_checked` is used once at load time; after
/// that the table is known good and `next` can be used by the interpreter.
pub struct NodeCursor<'m, 'a> {
    model: &'m Model<'a>,
    pos: usize,
    end: usize,
}

impl<'m, 'a> NodeCursor<'m, 'a> {
    pub fn new(model: &'m Model<'a>) -> Self {
        let pos = model.header.nodes_off as usize;
        NodeCursor {
            model,
            pos,
            end: pos + model.header.nodes_bytes as usize,
        }
    }

    pub fn next_checked(&mut self) -> Result<Option<Node<'a>>, Error> {
        if self.pos == self.end {
            return Ok(None);
        }
        if self.pos + NODE_HEADER_BYTES > self.end {
            return Err(Error::BadNode);
        }
        let b = self.model.raw;
        let opcode = rd_u16(b, self.pos);
        let n_in = b[self.pos + 2] as usize;
        let n_out = b[self.pos + 3] as usize;
        let attr_len = rd_u16(b, self.pos + 4) as usize;
        if attr_len % 4 != 0 || n_out == 0 {
            return Err(Error::BadNode);
        }
        let body = self.pos + NODE_HEADER_BYTES;
        let in_end = body + 4 * n_in;
        let out_end = in_end + 4 * n_out;
        let attr_end = out_end + attr_len;
        if attr_end > self.end {
            return Err(Error::BadNode);
        }
        let node = Node {
            opcode,
            raw_inputs: &b[body..in_end],
            raw_outputs: &b[in_end..out_end],
            raw_attrs: &b[out_end..attr_end],
        };
        self.pos = attr_end;
        Ok(Some(node))
    }

    pub fn next(&mut self) -> Option<Node<'a>> {
        self.next_checked().ok().flatten()
    }
}
