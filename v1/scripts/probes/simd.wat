;; Validated by js/src/detect.ts to decide whether this engine has baseline
;; 128-bit SIMD. Any instruction from the proposal would do; v128.const is
;; the smallest.
(module (func (export "p") (result v128) (v128.const i32x4 1 2 3 4)))
