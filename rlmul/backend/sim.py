"""Bit-parallel simulation of a generated multiplier netlist.

Every signal is held as one Python integer whose k-th bit is that signal's value
in test vector k, so a single ``&`` evaluates a gate across every vector at
once.  Python's arbitrary-precision integers make the vector count essentially
free, which is what lets an 8x8 multiplier be checked exhaustively (all 65536
input pairs in one pass) and a 16x16 one against millions of random vectors --
with no EDA tool installed.
"""

from __future__ import annotations

from collections.abc import Sequence

from .rtl import ZERO, build_netlist


def simulate(counts, width: int, a_values: Sequence[int], b_values: Sequence[int]) -> list[int]:
    """Return the netlist's product for each ``(a, b)`` pair."""
    if len(a_values) != len(b_values):
        raise ValueError("a_values and b_values must have the same length")
    n = len(a_values)
    net = build_netlist(counts, width)

    # Transpose the inputs into bit slices: a_bits[j] holds bit j of every vector.
    a_bits = [0] * width
    b_bits = [0] * width
    for k, (av, bv) in enumerate(zip(a_values, b_values)):
        for j in range(width):
            if (av >> j) & 1:
                a_bits[j] |= 1 << k
            if (bv >> j) & 1:
                b_bits[j] |= 1 << k

    sig = {ZERO: 0}
    for pp in net.partial_products:
        sig[pp.out] = a_bits[pp.j] & b_bits[pp.i]
    for cell in net.compressors:
        if cell.is_32:
            x, y, z = (sig[s] for s in cell.inputs)
            sig[cell.sum_out] = x ^ y ^ z
            sig[cell.carry_out] = (x & y) | (x & z) | (y & z)
        else:
            x, y = (sig[s] for s in cell.inputs)
            sig[cell.sum_out] = x ^ y
            sig[cell.carry_out] = x & y

    # Ripple-carry the two surviving rows, still bit-parallel.
    out_w = 2 * width
    carry = 0
    product_bits = []
    for c in range(out_w):
        x = sig[net.augend[c]] if c < len(net.augend) else 0
        y = sig[net.addend[c]] if c < len(net.addend) else 0
        product_bits.append(x ^ y ^ carry)
        carry = (x & y) | (x & carry) | (y & carry)

    products = []
    for k in range(n):
        value = 0
        for c in range(out_w):
            if (product_bits[c] >> k) & 1:
                value |= 1 << c
        products.append(value)
    return products


def check_exhaustive(counts, width: int) -> None:
    """Verify against every input pair. Only tractable for small widths."""
    if width > 8:
        raise ValueError(f"exhaustive check over 2**{2 * width} pairs is not tractable")
    a_values, b_values = [], []
    for a in range(1 << width):
        for b in range(1 << width):
            a_values.append(a)
            b_values.append(b)
    _assert_products(counts, width, a_values, b_values)


def check_random(counts, width: int, n_vectors: int = 100_000, seed: int = 0) -> None:
    """Verify against corner cases plus ``n_vectors`` random input pairs."""
    import random

    rng = random.Random(seed)
    hi = (1 << width) - 1
    # Clamped to `hi`: at tiny widths the fixed corners would otherwise exceed
    # the operand range, and simulate() would drop the extra bit while the
    # expected product kept it -- a false alarm on a correct netlist.
    corners = sorted({min(v, hi) for v in (0, 1, 2, hi, hi - 1, hi >> 1, (hi >> 1) + 1)})
    a_values = [a for a in corners for _ in corners]
    b_values = [b for _ in corners for b in corners]
    a_values += [rng.getrandbits(width) for _ in range(n_vectors)]
    b_values += [rng.getrandbits(width) for _ in range(n_vectors)]
    _assert_products(counts, width, a_values, b_values)


def _assert_products(counts, width, a_values, b_values) -> None:
    got = simulate(counts, width, a_values, b_values)
    for a, b, p in zip(a_values, b_values, got):
        if p != a * b:
            raise AssertionError(
                f"{width}x{width} netlist computed {a} * {b} = {p}, expected {a * b}"
            )
