"""Weight pre-packing and arena planning (design doc 6.1 and 8.1)."""

from __future__ import annotations

import numpy as np

MR, NR, MC = 4, 8, 64


def round_up(x: int, to: int) -> int:
    return (x + to - 1) // to * to


def pack_b(w: np.ndarray) -> np.ndarray:
    """Pack an (N, K) int8 weight matrix into panel layout v1.

    For each 8-column panel, for each group of 4 k, 8 columns x 4 bytes:

        [c0k0 c0k1 c0k2 c0k3 c1k0 ... c7k3][next k group] ...

    Doing this here means the runtime never packs weights — the cost is paid
    once at conversion instead of on every inference (doc 7.3).
    """
    n, k = w.shape
    kp, np_ = round_up(k, 4), round_up(n, NR)
    padded = np.zeros((np_, kp), dtype=np.int8)
    padded[:n, :k] = w
    # (np_, kp) -> panels of (NR, kp) -> (n_panels, kq, NR, 4)
    panels = padded.reshape(np_ // NR, NR, kp // 4, 4)
    return np.ascontiguousarray(panels.transpose(0, 2, 1, 3)).reshape(-1)


def packed_b_bytes(n: int, k: int) -> int:
    return round_up(n, NR) * round_up(k, 4)


def gemm_scratch_bytes(k: int) -> int:
    return MC * round_up(k, 4) + MC * 4


def plan_arena(nodes, tensors, align: int = 16) -> tuple[dict, int]:
    """Greedy best-fit arena allocation over activation lifetimes (doc 8.1).

    A tensor is live from the node that writes it to the last node that reads
    it; tensors whose lifetimes do not overlap share bytes. Graph inputs are
    live from the start and graph outputs to the end, since the caller writes
    and reads them outside the node loop.
    """
    acts = {n for n, t in tensors.items() if t.kind != "weight"}
    first_use: dict[str, int] = {}
    last_use: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for t in node.outputs:
            first_use.setdefault(t, i)
        for t in node.inputs:
            if t in acts:
                last_use[t] = i
    for name, t in tensors.items():
        if t.kind == "input":
            first_use[name] = -1
        if t.kind == "output":
            last_use[name] = len(nodes)

    free: list[list[int]] = []  # [offset, size], kept sorted by offset
    assign: dict[str, int] = {}
    high_water = 0

    def take(size: int) -> int:
        nonlocal high_water
        best = None
        for i, (off, sz) in enumerate(free):
            if sz >= size and (best is None or sz < free[best][1]):
                best = i
        if best is None:
            off = high_water
            high_water += size
            return off
        off, sz = free[best]
        if sz == size:
            free.pop(best)
        else:
            free[best] = [off + size, sz - size]
        return off

    def release(off: int, size: int) -> None:
        free.append([off, size])
        free.sort()
        merged: list[list[int]] = []
        for blk in free:
            if merged and merged[-1][0] + merged[-1][1] == blk[0]:
                merged[-1][1] += blk[1]
            else:
                merged.append(blk)
        free[:] = merged

    sizes = {n: round_up(tensors[n].nbytes, align) for n in acts}
    for name, t in tensors.items():
        if t.kind == "input":
            assign[name] = take(sizes[name])

    for i, node in enumerate(nodes):
        for t in node.outputs:
            if t in acts and t not in assign:
                assign[t] = take(sizes[t])
        for t in set(node.inputs):
            if t in acts and last_use.get(t) == i and tensors[t].kind != "output":
                release(assign[t], sizes[t])

    # Anything never consumed (a dead output, or a model with one node) still
    # needs a home.
    for name in acts:
        if name not in assign:
            assign[name] = take(sizes[name])
    return assign, high_water


def naive_arena_bytes(tensors) -> int:
    """What the same activations would cost with a buffer each — the
    'before' number in the README's memory comparison."""
    return sum(round_up(t.nbytes, 16) for t in tensors.values() if t.kind != "weight")
