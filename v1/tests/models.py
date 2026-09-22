"""ONNX models the test suite converts and runs.

Two kinds: single-operator graphs for per-op differential tests (doc 14.1),
and whole networks — a real trained MLP and a MobileNetV2-shaped CNN — for
end-to-end accuracy (doc 14.2).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _t(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def make_model(nodes, inputs, outputs, initializers, name="test") -> onnx.ModelProto:
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return onnx.shape_inference.infer_shapes(model)


def rng(seed=0):
    return np.random.default_rng(seed)


# ----------------------------------------------------------- single ops


def conv_model(n=1, c=3, h=8, w=8, out=4, k=3, stride=1, pad=1, dilation=1, group=1, act=None, seed=0, bias=True):
    r = rng(seed)
    cin = c // group
    wv = r.normal(scale=0.3, size=(out, cin, k, k)).astype(np.float32)
    inits = [numpy_helper.from_array(wv, "W")]
    ins = ["X", "W"]
    if bias:
        bv = r.normal(scale=0.1, size=(out,)).astype(np.float32)
        inits.append(numpy_helper.from_array(bv, "B"))
        ins.append("B")
    nodes = [helper.make_node(
        "Conv", ins, ["C"], kernel_shape=[k, k], strides=[stride, stride],
        pads=[pad, pad, pad, pad], dilations=[dilation, dilation], group=group, name="conv",
    )]
    last = "C"
    if act == "relu":
        nodes.append(helper.make_node("Relu", ["C"], ["R"], name="relu"))
        last = "R"
    elif act == "relu6":
        inits += [numpy_helper.from_array(np.float32(0.0), "lo"), numpy_helper.from_array(np.float32(6.0), "hi")]
        nodes.append(helper.make_node("Clip", ["C", "lo", "hi"], ["R"], name="clip"))
        last = "R"
    oh = (h + 2 * pad - dilation * (k - 1) - 1) // stride + 1
    ow = (w + 2 * pad - dilation * (k - 1) - 1) // stride + 1
    return make_model(nodes, [_t("X", (n, c, h, w))], [_t(last, (n, out, oh, ow))], inits)


def depthwise_model(c=8, h=7, w=7, k=3, stride=1, pad=1, seed=1):
    return conv_model(c=c, h=h, w=w, out=c, k=k, stride=stride, pad=pad, group=c, seed=seed)


def pool_model(kind="MaxPool", c=4, h=8, w=8, k=2, stride=2, pad=0, seed=2):
    if kind == "GlobalAveragePool":
        node = helper.make_node("GlobalAveragePool", ["X"], ["Y"], name="pool")
        oh = ow = 1
    else:
        node = helper.make_node(kind, ["X"], ["Y"], kernel_shape=[k, k], strides=[stride, stride], pads=[pad] * 4, name="pool")
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
    return make_model([node], [_t("X", (1, c, h, w))], [_t("Y", (1, c, oh, ow))], [])


def gemm_model(k=16, n=10, m=1, seed=3, act=None):
    r = rng(seed)
    wv = r.normal(scale=0.4, size=(n, k)).astype(np.float32)
    bv = r.normal(scale=0.1, size=(n,)).astype(np.float32)
    nodes = [helper.make_node("Gemm", ["X", "W", "B"], ["G"], transB=1, name="gemm")]
    last = "G"
    if act == "relu":
        nodes.append(helper.make_node("Relu", ["G"], ["R"], name="relu"))
        last = "R"
    return make_model(
        nodes, [_t("X", (m, k))], [_t(last, (m, n))],
        [numpy_helper.from_array(wv, "W"), numpy_helper.from_array(bv, "B")],
    )


def add_model(c=4, h=4, w=4, seed=4):
    r = rng(seed)
    w1 = r.normal(scale=0.3, size=(c, c, 1, 1)).astype(np.float32)
    w2 = r.normal(scale=0.7, size=(c, c, 1, 1)).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["X", "W1"], ["A"], kernel_shape=[1, 1], name="c1"),
        helper.make_node("Conv", ["X", "W2"], ["B"], kernel_shape=[1, 1], name="c2"),
        helper.make_node("Add", ["A", "B"], ["Y"], name="add"),
    ]
    return make_model(
        nodes, [_t("X", (1, c, h, w))], [_t("Y", (1, c, h, w))],
        [numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(w2, "W2")],
    )


def mul_model(c=4, h=3, w=3, seed=5):
    r = rng(seed)
    scale = r.normal(scale=1.0, size=(1, c, 1, 1)).astype(np.float32)
    nodes = [helper.make_node("Mul", ["X", "S"], ["Y"], name="mul")]
    return make_model(nodes, [_t("X", (1, c, h, w))], [_t("Y", (1, c, h, w))], [numpy_helper.from_array(scale, "S")])


def concat_model(c=3, h=4, w=4, seed=6):
    r = rng(seed)
    w1 = r.normal(scale=0.3, size=(c, c, 1, 1)).astype(np.float32)
    w2 = r.normal(scale=1.2, size=(c, c, 1, 1)).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["X", "W1"], ["A"], kernel_shape=[1, 1], name="c1"),
        helper.make_node("Conv", ["X", "W2"], ["B"], kernel_shape=[1, 1], name="c2"),
        helper.make_node("Concat", ["A", "B"], ["Y"], axis=1, name="cat"),
    ]
    return make_model(
        nodes, [_t("X", (1, c, h, w))], [_t("Y", (1, 2 * c, h, w))],
        [numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(w2, "W2")],
    )


def activation_model(kind="Sigmoid", c=4, h=3, w=3):
    node = helper.make_node(kind, ["X"], ["Y"], name="act")
    return make_model([node], [_t("X", (1, c, h, w))], [_t("Y", (1, c, h, w))], [])


def softmax_model(n=10, m=1):
    node = helper.make_node("Softmax", ["X"], ["Y"], axis=-1, name="softmax")
    return make_model([node], [_t("X", (m, n))], [_t("Y", (m, n))], [])


def batchnorm_model(c=4, h=5, w=5, seed=7):
    r = rng(seed)
    wv = r.normal(scale=0.3, size=(c, c, 3, 3)).astype(np.float32)
    gamma = np.abs(r.normal(1.0, 0.2, size=(c,))).astype(np.float32)
    beta = r.normal(0.0, 0.2, size=(c,)).astype(np.float32)
    mean = r.normal(0.0, 0.2, size=(c,)).astype(np.float32)
    var = np.abs(r.normal(1.0, 0.1, size=(c,))).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["X", "W"], ["C"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="conv"),
        helper.make_node("BatchNormalization", ["C", "g", "b", "m", "v"], ["N"], epsilon=1e-5, name="bn"),
        helper.make_node("Relu", ["N"], ["Y"], name="relu"),
    ]
    inits = [numpy_helper.from_array(a, n) for a, n in [(wv, "W"), (gamma, "g"), (beta, "b"), (mean, "m"), (var, "v")]]
    return make_model(nodes, [_t("X", (1, c, h, w))], [_t("Y", (1, c, h, w))], inits)


def flatten_fc_model(c=3, h=4, w=4, n=5, seed=8):
    """Flatten of a >1x1 feature map into a Gemm — the case where NHWC bytes
    and ONNX's NCHW flatten order disagree."""
    r = rng(seed)
    wv = r.normal(scale=0.2, size=(n, c * h * w)).astype(np.float32)
    bv = r.normal(scale=0.1, size=(n,)).astype(np.float32)
    nodes = [
        helper.make_node("Flatten", ["X"], ["F"], axis=1, name="flat"),
        helper.make_node("Gemm", ["F", "W", "B"], ["Y"], transB=1, name="gemm"),
    ]
    return make_model(
        nodes, [_t("X", (1, c, h, w))], [_t("Y", (1, n))],
        [numpy_helper.from_array(wv, "W"), numpy_helper.from_array(bv, "B")],
    )


# --------------------------------------------------------- whole networks


def mobilenet_block_model(c=16, h=16, w=16, expand=4, seed=9, classes=8):
    """A MobileNetV2-shaped network: pointwise expand, depthwise 3x3,
    pointwise project, residual add, global pool, classifier.

    Not the real weights — those are a download this environment cannot make
    — but the same topology, op mix and tensor shapes, which is what the
    kernels and the arena planner are being tested on.
    """
    r = rng(seed)
    inits, nodes = [], []

    def add_init(arr, name):
        inits.append(numpy_helper.from_array(arr.astype(np.float32), name))
        return name

    stem_w = r.normal(scale=0.25, size=(c, 3, 3, 3))
    add_init(stem_w, "stem_w")
    nodes.append(helper.make_node("Conv", ["X", "stem_w"], ["stem"], kernel_shape=[3, 3], pads=[1] * 4, strides=[2, 2], name="stem"))
    nodes.append(helper.make_node("Clip", ["stem", "zero", "six"], ["stem_a"], name="stem_relu6"))
    add_init(np.float32(0.0), "zero")
    add_init(np.float32(6.0), "six")
    cur, ch, hh, ww = "stem_a", c, h // 2, w // 2

    for i in range(2):
        e = ch * expand
        add_init(r.normal(scale=0.2, size=(e, ch, 1, 1)), f"exp{i}_w")
        nodes.append(helper.make_node("Conv", [cur, f"exp{i}_w"], [f"exp{i}"], kernel_shape=[1, 1], name=f"exp{i}"))
        nodes.append(helper.make_node("Clip", [f"exp{i}", "zero", "six"], [f"exp{i}a"], name=f"exp{i}relu"))

        add_init(r.normal(scale=0.3, size=(e, 1, 3, 3)), f"dw{i}_w")
        nodes.append(helper.make_node(
            "Conv", [f"exp{i}a", f"dw{i}_w"], [f"dw{i}"], kernel_shape=[3, 3],
            pads=[1] * 4, group=e, name=f"dw{i}",
        ))
        # A batch norm to fold, on the depthwise output where per-channel
        # scales matter most.
        add_init(np.abs(r.normal(1.0, 0.2, size=(e,))), f"dw{i}_g")
        add_init(r.normal(0.0, 0.1, size=(e,)), f"dw{i}_b")
        add_init(r.normal(0.0, 0.1, size=(e,)), f"dw{i}_m")
        add_init(np.abs(r.normal(1.0, 0.1, size=(e,))), f"dw{i}_v")
        nodes.append(helper.make_node(
            "BatchNormalization", [f"dw{i}", f"dw{i}_g", f"dw{i}_b", f"dw{i}_m", f"dw{i}_v"],
            [f"dw{i}n"], epsilon=1e-5, name=f"dw{i}bn",
        ))
        nodes.append(helper.make_node("Clip", [f"dw{i}n", "zero", "six"], [f"dw{i}a"], name=f"dw{i}relu"))

        add_init(r.normal(scale=0.2, size=(ch, e, 1, 1)), f"prj{i}_w")
        nodes.append(helper.make_node("Conv", [f"dw{i}a", f"prj{i}_w"], [f"prj{i}"], kernel_shape=[1, 1], name=f"prj{i}"))
        nodes.append(helper.make_node("Add", [cur, f"prj{i}"], [f"res{i}"], name=f"res{i}"))
        cur = f"res{i}"

    nodes.append(helper.make_node("GlobalAveragePool", [cur], ["pool"], name="pool"))
    nodes.append(helper.make_node("Flatten", ["pool"], ["flat"], axis=1, name="flat"))
    add_init(r.normal(scale=0.3, size=(classes, ch)), "fc_w")
    add_init(r.normal(scale=0.1, size=(classes,)), "fc_b")
    nodes.append(helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"], transB=1, name="fc"))
    nodes.append(helper.make_node("Softmax", ["logits"], ["Y"], axis=-1, name="softmax"))

    return make_model(nodes, [_t("X", (1, 3, h, w))], [_t("Y", (1, classes))], inits)


def train_digit_mlp(hidden=64, seed=0):
    """A real MLP trained on scikit-learn's 8x8 digits — actual learned
    weights, so the accuracy numbers in the README mean something."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    d = load_digits()
    x = (d.data / 16.0).astype(np.float32)
    y = d.target
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=600, random_state=seed, alpha=1e-3)
    clf.fit(xtr, ytr)

    w1 = clf.coefs_[0].T.astype(np.float32)
    b1 = clf.intercepts_[0].astype(np.float32)
    w2 = clf.coefs_[1].T.astype(np.float32)
    b2 = clf.intercepts_[1].astype(np.float32)
    nodes = [
        helper.make_node("Gemm", ["X", "W1", "B1"], ["h"], transB=1, name="fc1"),
        helper.make_node("Relu", ["h"], ["ha"], name="relu"),
        helper.make_node("Gemm", ["ha", "W2", "B2"], ["logits"], transB=1, name="fc2"),
        helper.make_node("Softmax", ["logits"], ["Y"], axis=-1, name="softmax"),
    ]
    inits = [numpy_helper.from_array(a, n) for a, n in [(w1, "W1"), (b1, "B1"), (w2, "W2"), (b2, "B2")]]
    model = make_model(nodes, [_t("X", (1, 64))], [_t("Y", (1, 10))], inits, name="digits-mlp")
    return model, (xtr, ytr, xte, yte), clf
