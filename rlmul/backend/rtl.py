"""Build and render a compressor-tree multiplier netlist.

The netlist is derived by simulating the dot matrix directly.  Each column owns
a list of one-bit signals; a compressor consumes signals from the front of its
column and appends its outputs to the back, the carry going to the next column
up.  Whatever is left when every compressor has fired becomes the two operands
of the final carry-propagate adder.

Consuming front-to-back and appending at the back is what makes the wiring
follow the compressor order from :func:`rlmul.ct.to_arch`, and it makes
correctness structural: every compressor preserves the weighted sum of the dots
it touches, so ``augend + addend`` is the product by construction.  Check that
independently with :mod:`rlmul.backend.sim`, which simulates the emitted netlist
against every input pair.

:func:`build_netlist` returns a tool-agnostic gate list so the same structure
feeds both the Verilog writer here and the simulator in :mod:`rlmul.backend.sim`.
"""

from __future__ import annotations

from typing import NamedTuple

from .. import ct

ZERO = "1'b0"


class PartialProduct(NamedTuple):
    """``out = a[j] & b[i]``"""
    out: str
    i: int
    j: int


class Compressor(NamedTuple):
    """A 3:2 (full adder) or 2:2 (half adder) cell."""
    sum_out: str
    carry_out: str
    inputs: tuple[str, ...]
    column: int
    stage: int

    @property
    def is_32(self) -> bool:
        return len(self.inputs) == 3


class Netlist(NamedTuple):
    width: int
    partial_products: list[PartialProduct]
    compressors: list[Compressor]
    augend: list[str]  # one signal per column, least significant first
    addend: list[str]
    n_stages: int


def build_netlist(counts, width: int) -> Netlist:
    """Wire up the compressor tree described by ``counts``."""
    stages = ct.assign_stages(counts, width)
    arch = ct.to_arch(stages, width)
    ncols = ct.n_columns(width)
    total_cols = ncols + 1  # the top column only ever receives carries

    # stage index of each compressor, in the same order as `arch`
    stage_of: list[int] = []
    for s in range(stages.shape[1]):
        stage_of += [s] * int(stages[0][s].sum() + stages[1][s].sum())

    cols: list[list[str]] = [[] for _ in range(total_cols)]
    head = [0] * total_cols  # first dot in each column not yet consumed

    pps: list[PartialProduct] = []
    for c in range(ncols):
        for i in range(max(0, c - width + 1), min(width - 1, c) + 1):
            pp = PartialProduct(f"pp_{i}_{c - i}", i, c - i)
            pps.append(pp)
            cols[c].append(pp.out)

    cells: list[Compressor] = []
    for idx, (col, is_32) in enumerate(arch):
        arity = 3 if is_32 else 2
        avail = len(cols[col]) - head[col]
        if avail < arity:
            raise ValueError(
                f"compressor {idx} wants {arity} dots from column {col} but only "
                f"{avail} are available; these counts are not schedulable"
            )
        ins = tuple(cols[col][head[col]:head[col] + arity])
        head[col] += arity
        cell = Compressor(f"s{idx}", f"c{idx}", ins, col, stage_of[idx])
        cells.append(cell)
        cols[col].append(cell.sum_out)
        cols[col + 1].append(cell.carry_out)

    augend, addend = [], []
    for c in range(total_cols):
        rest = cols[c][head[c]:]
        if len(rest) > 2:
            raise ValueError(
                f"column {c} still holds {len(rest)} dots after compression; "
                "the tree does not reduce to two rows"
            )
        augend.append(rest[0] if len(rest) >= 1 else ZERO)
        addend.append(rest[1] if len(rest) >= 2 else ZERO)

    return Netlist(width, pps, cells, augend, addend, int(stages.shape[1]))


def emit_verilog(counts, width: int, module_name: str = "MUL") -> str:
    """Render a flat combinational ``width x width`` unsigned multiplier.

    The output is ``2 * width`` bits wide -- the full product, not truncated.
    """
    net = build_netlist(counts, width)
    out_w = 2 * width

    def concat(bits: list[str]) -> str:
        # Verilog concatenation is most-significant-first.
        return "{" + ", ".join(reversed(bits)) + "}"

    body = ["  // partial products: a[j] & b[i] has weight 2**(i+j)"]
    body += [f"  wire {pp.out} = a[{pp.j}] & b[{pp.i}];" for pp in net.partial_products]

    if net.compressors:
        body.append("")
        body.append(
            f"  // {len(net.compressors)} compressors in {net.n_stages} reduction stages"
        )
    stage = -1
    for cell in net.compressors:
        if cell.stage != stage:
            stage = cell.stage
            body.append(f"  // -- stage {stage}")
        if cell.is_32:
            x, y, z = cell.inputs
            body.append(f"  wire {cell.sum_out} = {x} ^ {y} ^ {z};")
            body.append(f"  wire {cell.carry_out} = ({x} & {y}) | ({x} & {z}) | ({y} & {z});")
        else:
            x, y = cell.inputs
            body.append(f"  wire {cell.sum_out} = {x} ^ {y};")
            body.append(f"  wire {cell.carry_out} = {x} & {y};")

    lines = [
        f"// {width}x{width} unsigned multiplier, {len(net.compressors)} compressors,",
        f"// {net.n_stages} reduction stages. Generated by rlmul -- do not edit.",
        f"module {module_name} (",
        f"  input  wire [{width - 1}:0] a,",
        f"  input  wire [{width - 1}:0] b,",
        f"  output wire [{out_w - 1}:0] p",
        ");",
        *body,
        "",
        "  // final carry-propagate adder over the two surviving rows",
        f"  wire [{out_w - 1}:0] augend = {concat(net.augend)};",
        f"  wire [{out_w - 1}:0] addend = {concat(net.addend)};",
        "  assign p = augend + addend;",
        "endmodule",
        "",
    ]
    return "\n".join(lines)
