//! The interpreter (design doc 13.3) and the per-operator validator.
//!
//! `validate` runs once at load and checks everything a kernel will later
//! assume: operand counts, dtypes, which operands are weights, shapes,
//! attribute ranges and scratch size. After it passes, `run` executes the
//! node table with raw pointer arithmetic and no further checks. That split
//! is what lets the hot path be fast *and* keeps a hostile model file from
//! turning into an out-of-bounds read.

use crate::error::Error;
use crate::kernels::conv::{self, ConvArgs, ConvGeom};
use crate::kernels::elementwise as ew;
use crate::kernels::gemm::{self, Gemm, Isa};
use crate::model::*;

pub const OP_CONV2D: u16 = 0;
pub const OP_DEPTHWISE_CONV2D: u16 = 1;
pub const OP_FULLY_CONNECTED: u16 = 2;
pub const OP_MATMUL: u16 = 3;
pub const OP_ADD: u16 = 4;
pub const OP_MUL: u16 = 5;
pub const OP_MAXPOOL2D: u16 = 6;
pub const OP_AVGPOOL2D: u16 = 7;
pub const OP_SOFTMAX: u16 = 8;
pub const OP_RESHAPE: u16 = 9;
pub const OP_TRANSPOSE: u16 = 10;
pub const OP_CONCAT: u16 = 11;
pub const OP_CLAMP: u16 = 12;
pub const OP_LOOKUP: u16 = 13;
pub const OP_LAYERNORM: u16 = 14;
pub const OP_QUANTIZE: u16 = 15;
pub const OP_DEQUANTIZE: u16 = 16;

/// Everything `run` needs besides the model.
pub struct Buffers {
    pub arena: *mut u8,
    pub scratch: *mut u8,
    pub isa: Isa,
}

impl<'a> Model<'a> {
    #[inline(always)]
    fn ptr(&self, t: &Tensor, b: &Buffers) -> *mut u8 {
        if t.is_activation() {
            unsafe { b.arena.add(t.offset as usize) }
        } else {
            unsafe { self.weights().as_ptr().add(t.offset as usize) as *mut u8 }
        }
    }
}

fn need(cond: bool, e: Error) -> Result<(), Error> {
    if cond {
        Ok(())
    } else {
        Err(e)
    }
}

fn is_i8_act(t: &Tensor) -> bool {
    t.is_activation() && t.dtype == DTYPE_I8 && t.n_scales >= 1
}

fn is_weight(t: &Tensor, dtype: u8, min_elems: usize) -> bool {
    t.kind == KIND_WEIGHT && t.dtype == dtype && t.numel() >= min_elems
}

fn clamp_attr_ok(lo: i32, hi: i32) -> bool {
    -128 <= lo && lo <= hi && hi <= 127
}

fn shift_ok(s: i32) -> bool {
    (-31..=30).contains(&s)
}

/// Requant parameter arrays: `mult` must be a valid Q31 value (≥ 0) and
/// every `shift` in range, or the fixed-point code misbehaves.
fn requant_params_ok(m: &Model, mult: &Tensor, shift: &Tensor, n: usize) -> bool {
    let w = m.weights();
    for i in 0..n {
        let mo = mult.offset as usize + 4 * i;
        let so = shift.offset as usize + 4 * i;
        let mv = i32::from_le_bytes([w[mo], w[mo + 1], w[mo + 2], w[mo + 3]]);
        let sv = i32::from_le_bytes([w[so], w[so + 1], w[so + 2], w[so + 3]]);
        if mv < 0 || !shift_ok(sv) {
            return false;
        }
    }
    true
}

fn conv_geom(node: &Node, x: &Tensor, y: &Tensor, kh: usize, kw: usize) -> Result<ConvGeom, Error> {
    let (h, w, c) = x.hwc();
    let (oh, ow, _) = y.hwc();
    let sh = node.attr(2);
    let sw = node.attr(3);
    let pads = [node.attr(4), node.attr(5), node.attr(6), node.attr(7)];
    let dh = node.attr(8);
    let dw = node.attr(9);
    need(sh >= 1 && sw >= 1 && sh <= 64 && sw <= 64, Error::BadAttr)?;
    need(dh >= 1 && dw >= 1 && dh <= 64 && dw <= 64, Error::BadAttr)?;
    need(pads.iter().all(|&p| (0..=64).contains(&p)), Error::BadAttr)?;
    need(kh >= 1 && kw >= 1 && kh <= 64 && kw <= 64, Error::BadAttr)?;
    let g = ConvGeom {
        h,
        w,
        c,
        kh,
        kw,
        sh: sh as usize,
        sw: sw as usize,
        pt: pads[0] as usize,
        pl: pads[1] as usize,
        dh: dh as usize,
        dw: dw as usize,
        oh,
        ow,
    };
    let eh = conv::out_extent(h, pads[0] as usize, pads[2] as usize, kh, g.sh, g.dh)
        .ok_or(Error::BadShape)?;
    let ew_ = conv::out_extent(w, pads[1] as usize, pads[3] as usize, kw, g.sw, g.dw)
        .ok_or(Error::BadShape)?;
    need(eh == oh && ew_ == ow, Error::BadShape)?;
    need(
        x.rank == 4 && y.rank == 4 && x.dims[0] == 1 && y.dims[0] == 1,
        Error::BadShape,
    )?;
    Ok(g)
}

/// Check one node against everything its kernel will assume.
pub fn validate_node(m: &Model, node: &Node) -> Result<(), Error> {
    let t = |i: usize| m.tensor(node.input(i));
    let out = m.tensor(node.output(0));
    let scratch = m.header.scratch_bytes as usize;

    match node.opcode {
        OP_CONV2D | OP_DEPTHWISE_CONV2D => {
            need(
                node.n_in() == 5 && node.n_out() == 1 && node.n_attrs() >= 12,
                Error::BadNode,
            )?;
            let (x, w, bias, mult, shift) = (t(0), t(1), t(2), t(3), t(4));
            need(is_i8_act(&x) && is_i8_act(&out), Error::BadDtype)?;
            need(
                w.kind == KIND_WEIGHT && w.dtype == DTYPE_I8 && w.rank == 4,
                Error::BadDtype,
            )?;
            let lo = node.attr(10);
            let hi = node.attr(11);
            need(clamp_attr_ok(lo, hi), Error::BadAttr)?;
            let (kh, kw) = (node.attr(0), node.attr(1));
            need(kh >= 1 && kw >= 1, Error::BadAttr)?;
            let g = conv_geom(node, &x, &out, kh as usize, kw as usize)?;
            let n = out.last_dim();
            need(
                is_weight(&bias, DTYPE_I32, n)
                    && is_weight(&mult, DTYPE_I32, n)
                    && is_weight(&shift, DTYPE_I32, n),
                Error::BadDtype,
            )?;
            need(requant_params_ok(m, &mult, &shift, n), Error::BadAttr)?;
            if node.opcode == OP_CONV2D {
                // w logical dims (N, KH, KW, C), packed.
                need(
                    w.dims[0] as usize == n
                        && w.dims[1] as usize == g.kh
                        && w.dims[2] as usize == g.kw
                        && w.dims[3] as usize == g.c,
                    Error::BadShape,
                )?;
                need(
                    w.nbytes as usize >= gemm::packed_b_bytes(n, g.k()),
                    Error::BadShape,
                )?;
                let req = gemm::round_up(g.im2col_bytes(), 16) + gemm::gemm_scratch_bytes(g.k());
                need(req <= scratch, Error::ScratchTooSmall)?;
            } else {
                // depthwise: w dims (1, KH, KW, C), channel multiplier 1.
                need(n == g.c, Error::BadShape)?;
                need(
                    w.dims[1] as usize == g.kh
                        && w.dims[2] as usize == g.kw
                        && w.dims[3] as usize == g.c,
                    Error::BadShape,
                )?;
                need(w.nbytes as usize >= g.kh * g.kw * g.c, Error::BadShape)?;
            }
        }
        OP_FULLY_CONNECTED => {
            need(
                node.n_in() == 5 && node.n_out() == 1 && node.n_attrs() >= 2,
                Error::BadNode,
            )?;
            let (x, w, bias, mult, shift) = (t(0), t(1), t(2), t(3), t(4));
            need(is_i8_act(&x) && is_i8_act(&out), Error::BadDtype)?;
            need(
                w.kind == KIND_WEIGHT && w.dtype == DTYPE_I8 && w.rank == 2,
                Error::BadDtype,
            )?;
            let (n, k) = (w.dims[0] as usize, w.dims[1] as usize);
            need(x.numel() % k == 0, Error::BadShape)?;
            let rows = x.numel() / k;
            need(
                out.numel() == rows * n && out.last_dim() == n,
                Error::BadShape,
            )?;
            need(
                w.nbytes as usize >= gemm::packed_b_bytes(n, k),
                Error::BadShape,
            )?;
            need(
                is_weight(&bias, DTYPE_I32, n)
                    && is_weight(&mult, DTYPE_I32, n)
                    && is_weight(&shift, DTYPE_I32, n),
                Error::BadDtype,
            )?;
            need(requant_params_ok(m, &mult, &shift, n), Error::BadAttr)?;
            need(clamp_attr_ok(node.attr(0), node.attr(1)), Error::BadAttr)?;
            need(
                gemm::gemm_scratch_bytes(k) <= scratch,
                Error::ScratchTooSmall,
            )?;
        }
        OP_ADD => {
            need(
                node.n_in() == 2 && node.n_out() == 1 && node.n_attrs() >= 9,
                Error::BadNode,
            )?;
            let (a, b) = (t(0), t(1));
            need(
                is_i8_act(&a) && is_i8_act(&b) && is_i8_act(&out),
                Error::BadDtype,
            )?;
            need(
                a.numel() == out.numel() && b.numel() == out.numel(),
                Error::BadShape,
            )?;
            need((0..=22).contains(&node.attr(0)), Error::BadAttr)?;
            for i in [1, 3, 5] {
                need(
                    node.attr(i) >= 0 && shift_ok(node.attr(i + 1)),
                    Error::BadAttr,
                )?;
            }
            need(clamp_attr_ok(node.attr(7), node.attr(8)), Error::BadAttr)?;
        }
        OP_MUL => {
            need(
                node.n_in() == 2 && node.n_out() == 1 && node.n_attrs() >= 4,
                Error::BadNode,
            )?;
            let (a, b) = (t(0), t(1));
            need(is_i8_act(&a) && is_i8_act(&out), Error::BadDtype)?;
            // The second operand may be a constant (a weight) or an activation.
            need(b.dtype == DTYPE_I8 && b.n_scales >= 1, Error::BadDtype)?;
            need(a.numel() == out.numel(), Error::BadShape)?;
            let nb = b.numel();
            need(
                nb == a.numel() || nb == 1 || nb == a.last_dim(),
                Error::BadShape,
            )?;
            need(node.attr(0) >= 0 && shift_ok(node.attr(1)), Error::BadAttr)?;
            need(clamp_attr_ok(node.attr(2), node.attr(3)), Error::BadAttr)?;
        }
        OP_MAXPOOL2D | OP_AVGPOOL2D => {
            need(
                node.n_in() == 1 && node.n_out() == 1 && node.n_attrs() >= 10,
                Error::BadNode,
            )?;
            let x = t(0);
            need(is_i8_act(&x) && is_i8_act(&out), Error::BadDtype)?;
            need(x.last_dim() == out.last_dim(), Error::BadShape)?;
            let (kh, kw) = (node.attr(0), node.attr(1));
            need(
                kh >= 1 && kw >= 1 && kh <= 4096 && kw <= 4096,
                Error::BadAttr,
            )?;
            // Pools have no dilation; reuse the conv geometry check with 1s.
            let (h, w, c) = x.hwc();
            let (oh, ow, _) = out.hwc();
            let (sh, sw) = (node.attr(2), node.attr(3));
            let pads = [node.attr(4), node.attr(5), node.attr(6), node.attr(7)];
            need(
                sh >= 1 && sw >= 1 && pads.iter().all(|&p| (0..=4096).contains(&p)),
                Error::BadAttr,
            )?;
            let eh = conv::out_extent(
                h,
                pads[0] as usize,
                pads[2] as usize,
                kh as usize,
                sh as usize,
                1,
            )
            .ok_or(Error::BadShape)?;
            let ew_ = conv::out_extent(
                w,
                pads[1] as usize,
                pads[3] as usize,
                kw as usize,
                sw as usize,
                1,
            )
            .ok_or(Error::BadShape)?;
            need(
                eh == oh && ew_ == ow && x.numel() == h * w * c,
                Error::BadShape,
            )?;
            need(clamp_attr_ok(node.attr(8), node.attr(9)), Error::BadAttr)?;
        }
        OP_SOFTMAX => {
            need(node.n_in() == 2 && node.n_out() == 1, Error::BadNode)?;
            let (x, table) = (t(0), t(1));
            need(is_i8_act(&x) && is_i8_act(&out), Error::BadDtype)?;
            need(is_weight(&table, DTYPE_F32, 256), Error::BadDtype)?;
            need(x.numel() == out.numel(), Error::BadShape)?;
        }
        OP_RESHAPE => {
            need(node.n_in() == 1 && node.n_out() == 1, Error::BadNode)?;
            let x = t(0);
            need(x.is_activation() && x.dtype == out.dtype, Error::BadDtype)?;
            need(x.numel() == out.numel(), Error::BadShape)?;
        }
        OP_CONCAT => {
            let n = node.n_in();
            need(
                n >= 1 && node.n_out() == 1 && node.n_attrs() >= 2 * n,
                Error::BadNode,
            )?;
            need(is_i8_act(&out), Error::BadDtype)?;
            let c_out = out.last_dim();
            let outer = out.numel() / c_out;
            let mut total = 0usize;
            for i in 0..n {
                let x = t(i);
                need(is_i8_act(&x), Error::BadDtype)?;
                let c = x.last_dim();
                need(x.numel() == outer * c, Error::BadShape)?;
                need(
                    node.attr(2 * i) >= 0 && shift_ok(node.attr(2 * i + 1)),
                    Error::BadAttr,
                )?;
                total += c;
            }
            need(total == c_out, Error::BadShape)?;
        }
        OP_CLAMP => {
            need(
                node.n_in() == 1 && node.n_out() == 1 && node.n_attrs() >= 2,
                Error::BadNode,
            )?;
            let x = t(0);
            need(
                is_i8_act(&x) && is_i8_act(&out) && x.numel() == out.numel(),
                Error::BadShape,
            )?;
            need(clamp_attr_ok(node.attr(0), node.attr(1)), Error::BadAttr)?;
        }
        OP_LOOKUP => {
            need(node.n_in() == 2 && node.n_out() == 1, Error::BadNode)?;
            let (x, table) = (t(0), t(1));
            need(
                is_i8_act(&x) && is_i8_act(&out) && x.numel() == out.numel(),
                Error::BadShape,
            )?;
            need(is_weight(&table, DTYPE_I8, 256), Error::BadDtype)?;
        }
        OP_QUANTIZE => {
            need(node.n_in() == 1 && node.n_out() == 1, Error::BadNode)?;
            let x = t(0);
            need(
                x.is_activation() && x.dtype == DTYPE_F32 && is_i8_act(&out),
                Error::BadDtype,
            )?;
            need(x.numel() == out.numel(), Error::BadShape)?;
        }
        OP_DEQUANTIZE => {
            need(node.n_in() == 1 && node.n_out() == 1, Error::BadNode)?;
            let x = t(0);
            need(is_i8_act(&x) && out.dtype == DTYPE_F32, Error::BadDtype)?;
            need(x.numel() == out.numel(), Error::BadShape)?;
        }
        // MatMul (activation × activation), Transpose and LayerNorm are
        // defined in the format but not implemented in v1; the converter
        // refuses models that need them rather than emitting them.
        _ => return Err(Error::UnsupportedOp),
    }
    Ok(())
}

pub fn validate(m: &Model) -> Result<(), Error> {
    let mut c = NodeCursor::new(m);
    while let Some(node) = c.next_checked()? {
        validate_node(m, &node)?;
    }
    Ok(())
}

/// Execute the graph. Performs no allocation.
///
/// # Safety
/// `m` must have passed [`validate`], and `b` must point at an arena of
/// `arena_bytes` and a 16-byte-aligned scratch of `scratch_bytes`.
pub unsafe fn run(m: &Model, b: &Buffers) {
    let mut cursor = NodeCursor::new(m);
    while let Some(node) = cursor.next() {
        let t = |i: usize| m.tensor(node.input(i));
        let out = m.tensor(node.output(0));
        let y = m.ptr(&out, b);
        match node.opcode {
            OP_CONV2D | OP_DEPTHWISE_CONV2D => {
                let (x, w) = (t(0), t(1));
                let g = conv_geom(
                    &node,
                    &x,
                    &out,
                    node.attr(0) as usize,
                    node.attr(1) as usize,
                )
                .ok()
                .unwrap_unchecked();
                let args = ConvArgs {
                    x: m.ptr(&x, b) as *const i8,
                    w: m.ptr(&w, b) as *const i8,
                    bias: m.ptr(&t(2), b) as *const i32,
                    mult: m.ptr(&t(3), b) as *const i32,
                    shift: m.ptr(&t(4), b) as *const i32,
                    y: y as *mut i8,
                    n: out.last_dim(),
                    x_zero_point: x.zero_point,
                    y_zero_point: out.zero_point,
                    lo: node.attr(10),
                    hi: node.attr(11),
                };
                if node.opcode == OP_CONV2D {
                    conv::conv2d(&g, &args, b.scratch, b.isa);
                } else {
                    conv::depthwise(&g, &args, b.isa);
                }
            }
            OP_FULLY_CONNECTED => {
                let (x, w) = (t(0), t(1));
                let (n, k) = (w.dims[0] as usize, w.dims[1] as usize);
                let g = Gemm {
                    m: x.numel() / k,
                    n,
                    k,
                    a: m.ptr(&x, b) as *const i8,
                    lda: k,
                    b: m.ptr(&w, b) as *const i8,
                    bias: m.ptr(&t(2), b) as *const i32,
                    mult: m.ptr(&t(3), b) as *const i32,
                    shift: m.ptr(&t(4), b) as *const i32,
                    out: y as *mut i8,
                    ldc: n,
                    zero_point: out.zero_point,
                    lo: node.attr(0),
                    hi: node.attr(1),
                };
                gemm::gemm(&g, b.scratch, b.isa);
            }
            OP_ADD => {
                let (a1, a2) = (t(0), t(1));
                let p = ew::AddParams {
                    left_shift: node.attr(0),
                    m1: node.attr(1),
                    s1: node.attr(2),
                    m2: node.attr(3),
                    s2: node.attr(4),
                    mo: node.attr(5),
                    so: node.attr(6),
                    z1: a1.zero_point,
                    z2: a2.zero_point,
                    zo: out.zero_point,
                    lo: node.attr(7),
                    hi: node.attr(8),
                };
                ew::add(
                    m.ptr(&a1, b) as *const i8,
                    m.ptr(&a2, b) as *const i8,
                    y as *mut i8,
                    out.numel(),
                    &p,
                );
            }
            OP_MUL => {
                let (a1, a2) = (t(0), t(1));
                ew::mul(
                    m.ptr(&a1, b) as *const i8,
                    m.ptr(&a2, b) as *const i8,
                    y as *mut i8,
                    out.numel(),
                    a2.numel(),
                    a1.zero_point,
                    a2.zero_point,
                    out.zero_point,
                    node.attr(0),
                    node.attr(1),
                    node.attr(2),
                    node.attr(3),
                );
            }
            OP_MAXPOOL2D | OP_AVGPOOL2D => {
                let x = t(0);
                let (h, w, c) = x.hwc();
                let (oh, ow, _) = out.hwc();
                let g = ConvGeom {
                    h,
                    w,
                    c,
                    kh: node.attr(0) as usize,
                    kw: node.attr(1) as usize,
                    sh: node.attr(2) as usize,
                    sw: node.attr(3) as usize,
                    pt: node.attr(4) as usize,
                    pl: node.attr(5) as usize,
                    dh: 1,
                    dw: 1,
                    oh,
                    ow,
                };
                conv::pool(
                    &g,
                    m.ptr(&x, b) as *const i8,
                    y as *mut i8,
                    node.opcode == OP_AVGPOOL2D,
                    node.attr(8),
                    node.attr(9),
                );
            }
            OP_SOFTMAX => {
                let x = t(0);
                let c = x.last_dim();
                ew::softmax(
                    m.ptr(&x, b) as *const i8,
                    m.ptr(&t(1), b) as *const f32,
                    y as *mut i8,
                    x.numel() / c,
                    c,
                );
            }
            OP_RESHAPE => {
                let x = t(0);
                let src = m.ptr(&x, b);
                if src != y {
                    // `copy`, not `copy_nonoverlapping`: the arena planner
                    // is free to overlap a dead input with its output.
                    core::ptr::copy(src, y, x.numel() * x.elem_bytes());
                }
            }
            OP_CONCAT => {
                let c_out = out.last_dim();
                let outer = out.numel() / c_out;
                let mut off = 0;
                for i in 0..node.n_in() {
                    let x = t(i);
                    let c = x.last_dim();
                    ew::concat_input(
                        m.ptr(&x, b) as *const i8,
                        y as *mut i8,
                        outer,
                        c,
                        c_out,
                        off,
                        x.zero_point,
                        out.zero_point,
                        node.attr(2 * i),
                        node.attr(2 * i + 1),
                    );
                    off += c;
                }
            }
            OP_CLAMP => {
                let x = t(0);
                ew::clamp_i8(
                    m.ptr(&x, b) as *const i8,
                    y as *mut i8,
                    x.numel(),
                    node.attr(0),
                    node.attr(1),
                );
            }
            OP_LOOKUP => {
                let x = t(0);
                ew::lookup(
                    m.ptr(&x, b) as *const i8,
                    m.ptr(&t(1), b) as *const i8,
                    y as *mut i8,
                    x.numel(),
                );
            }
            OP_QUANTIZE => {
                let x = t(0);
                ew::quantize(
                    m.ptr(&x, b) as *const f32,
                    y as *mut i8,
                    x.numel(),
                    m.scale(&out),
                    out.zero_point,
                );
            }
            OP_DEQUANTIZE => {
                let x = t(0);
                ew::dequantize(
                    m.ptr(&x, b) as *const i8,
                    y as *mut f32,
                    x.numel(),
                    m.scale(&x),
                    x.zero_point,
                );
            }
            _ => core::hint::unreachable_unchecked(),
        }
    }
}
