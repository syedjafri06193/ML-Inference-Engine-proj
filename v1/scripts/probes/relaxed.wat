;; The relaxed-SIMD probe: the int8 dot product itself, which is the only
;; instruction from that proposal this runtime actually needs.
(module (func (export "p") (param v128 v128 v128) (result v128)
  (i32x4.relaxed_dot_i8x16_i7x16_add_s (local.get 0) (local.get 1) (local.get 2))))
