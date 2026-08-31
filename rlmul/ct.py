"""Compressor-tree state space for unsigned multiplier design.

A ``W x W`` unsigned multiplier starts from an AND array of partial products:
partial product ``a[j] & b[i]`` has weight ``2**(i+j)``, so column ``c`` of the
dot matrix initially holds ``initial_dots(W)[c]`` dots.  A compressor tree
reduces every column to at most two dots using 3:2 compressors (full adders)
and 2:2 compressors (half adders); the two surviving rows are then summed by a
final carry-propagate adder.

The RL state is the per-column compressor count::

    counts[0, c] = number of 3:2 compressors in column c
    counts[1, c] = number of 2:2 compressors in column c

for ``c`` in ``[0, 2W-1)``.  Which reduction stage each compressor lands in is
*derived* from the counts by :func:`assign_stages`, so the state stays small
and the agent only has to reason about column-wise structure.

Every function here is pure: arguments are never mutated.
"""

from __future__ import annotations

import numpy as np

# Action encoding: action = column * 4 + kind
ADD_22 = 0  # add one 2:2 compressor
DEL_22 = 1  # remove one 2:2 compressor
_32_TO_22 = 2  # replace one 3:2 by one 2:2
_22_TO_32 = 3  # replace one 2:2 by one 3:2
N_ACTION_KINDS = 4


def n_columns(width: int) -> int:
    """Number of dot-matrix columns that can hold partial products."""
    return 2 * width - 1


def n_actions(width: int) -> int:
    return n_columns(width) * N_ACTION_KINDS


def initial_dots(width: int) -> np.ndarray:
    """Dots per column in the raw AND array, shape ``(2W-1,)``.

    Column ``c`` is fed by every ``(i, j)`` with ``i + j == c``, which gives the
    familiar triangle ``1, 2, ..., W, ..., 2, 1``.
    """
    cols = np.arange(n_columns(width))
    return np.where(cols < width, cols + 1, 2 * width - 1 - cols).astype(np.int64)


def remaining_dots(counts: np.ndarray, width: int) -> np.ndarray:
    """Dots left in each column once every compressor has fired.

    Returns an array of length ``2W``: the extra entry is the carry-out of the
    most significant partial-product column, which the final adder still has to
    absorb.  A 3:2 compressor turns 3 dots into 1 sum plus a carry into the next
    column (net ``-2`` locally, ``+1`` next); a 2:2 turns 2 into 1 plus a carry
    (net ``-1`` locally, ``+1`` next).
    """
    counts = np.asarray(counts, dtype=np.int64)
    ncols = n_columns(width)
    dots = np.zeros(ncols + 1, dtype=np.int64)
    dots[:ncols] = initial_dots(width)
    carries = counts[0] + counts[1]  # one carry out of every compressor
    dots[1:] += carries  # carry from column c lands in column c+1
    dots[:ncols] -= 2 * counts[0] + counts[1]
    return dots


def is_legal(counts: np.ndarray, width: int) -> bool:
    """True when the tree reduces to two rows the final adder can sum.

    Every partial-product column must end with one or two dots.  The extra
    carry-out column above them may also be empty -- nothing has to reach it --
    but it cannot hold three, because there is no third row to put them in and
    no further column to compress them into.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if (counts < 0).any():
        return False
    dots = remaining_dots(counts, width)
    body, carry_out = dots[:-1], dots[-1]
    return bool(((body >= 1) & (body <= 2)).all() and 0 <= carry_out <= 2)


class IllegalAction(ValueError):
    """Raised when an edit cannot be legalised into a buildable tree."""


def apply_action(counts: np.ndarray, width: int, action: int) -> np.ndarray:
    """Apply one edit and re-legalise the columns above it.

    The edit changes the dot balance of column ``col`` (and its neighbour);
    columns to the left are untouched, so legality is restored by sweeping
    right, inserting or deleting compressors until every column is back to one
    or two dots.  Returns a new array; ``counts`` is left alone.

    Raises :class:`IllegalAction` when the sweep cannot settle -- in particular
    when it pushes a third carry into the column above the partial products,
    which the two-row final adder has nowhere to put.  :func:`legal_action_mask`
    filters those out, so a masked action never raises.
    """
    result = _try_apply(counts, width, action)
    if result is None:
        raise IllegalAction(
            f"action {action} on width {width} cannot be legalised"
        )
    return result


def _try_apply(counts: np.ndarray, width: int, action: int) -> np.ndarray | None:
    counts = np.asarray(counts, dtype=np.int64).copy()
    ncols = n_columns(width)
    col, kind = divmod(int(action), N_ACTION_KINDS)
    dots = remaining_dots(counts, width)

    if kind == ADD_22:
        counts[1][col] += 1
        dots[col] -= 1
        dots[col + 1] += 1
    elif kind == DEL_22:
        counts[1][col] -= 1
        dots[col] += 1
        dots[col + 1] -= 1
    elif kind == _32_TO_22:
        counts[1][col] += 1
        counts[0][col] -= 1
        dots[col] += 1
    elif kind == _22_TO_32:
        counts[1][col] -= 1
        counts[0][col] += 1
        dots[col] -= 1
    else:
        raise ValueError(f"unknown action kind {kind}")

    for c in range(col + 1, ncols):
        if dots[c] in (1, 2):
            break
        if dots[c] == 3:
            # one more 3:2 soaks up the surplus
            counts[0][c] += 1
            dots[c] = 1
            dots[c + 1] += 1
        elif dots[c] == 0:
            # give a dot back by retiring a compressor
            if counts[1][c] >= 1:
                counts[1][c] -= 1
                dots[c] = 1
            else:
                counts[0][c] -= 1
                dots[c] = 2
            dots[c + 1] -= 1
        else:
            return None
    # The sweep starts above `col`, so the edited column itself has never been
    # re-checked, and neither has anything the sweep stopped short of --
    # validate the whole dot profile before returning.
    if (counts < 0).any():
        return None
    if not ((dots[:ncols] >= 1) & (dots[:ncols] <= 2)).all():
        return None
    if not 0 <= dots[ncols] <= 2:
        return None
    return counts


def legal_action_mask(counts: np.ndarray, width: int,
                      max_stages: int | None = None) -> np.ndarray:
    """Boolean mask over ``n_actions(width)`` marking applicable edits.

    An edit is applicable when the column it targets has the right dot balance
    and holds the compressor the edit wants to consume.  When ``max_stages`` is
    given, edits whose result would need more reduction stages than the state
    encoding can represent are masked out too -- that check needs a full
    transition plus stage assignment, so it is skipped when not required.
    """
    counts = np.asarray(counts, dtype=np.int64)
    ncols = n_columns(width)
    dots = remaining_dots(counts, width)
    mask = np.zeros(n_actions(width), dtype=bool)

    # Edits are confined to columns 1 .. 2W-3: column 0 always holds exactly
    # one dot, so nothing can be done to it, and the top partial-product
    # column, 2W-2, is outside the action space.
    for c in range(1, ncols - 1):
        if dots[c] == 2:
            mask[c * N_ACTION_KINDS + ADD_22] = True
            if counts[1][c] >= 1:
                mask[c * N_ACTION_KINDS + _22_TO_32] = True
        elif dots[c] == 1:
            if counts[0][c] >= 1:
                mask[c * N_ACTION_KINDS + _32_TO_22] = True
            if counts[1][c] >= 1:
                mask[c * N_ACTION_KINDS + DEL_22] = True

    # The dot balance above is necessary but not sufficient: the legalisation
    # sweep can fail to settle, or can push a third carry into the top column.
    # Simulating the sweep is cheap, so it is always checked.
    for action in np.flatnonzero(mask):
        nxt = _try_apply(counts, width, int(action))
        if nxt is None:
            mask[action] = False
        elif max_stages is not None and num_stages(nxt, width) > max_stages:
            mask[action] = False
    return mask


def assign_stages(counts: np.ndarray, width: int) -> np.ndarray:
    """Spread the per-column compressor counts over reduction stages.

    Greedy, earliest-stage-first: a column's compressors are pushed into the
    first stage whose available dots can feed them, and whatever does not fit
    spills into the next stage.  Returns an array of shape ``(2, S, 2W-1)``
    where ``[0]`` counts 3:2 and ``[1]`` counts 2:2 compressors per stage and
    column.
    """
    counts = np.asarray(counts, dtype=np.int64)
    ncols = n_columns(width)
    left = counts.copy()  # compressors still waiting for a stage

    stage32 = [np.zeros(ncols, dtype=np.int64)]
    stage22 = [np.zeros(ncols, dtype=np.int64)]
    avail = [initial_dots(width).copy()]  # dots each stage can feed, per column

    # A stage can only fire compressors that carries from the column below have
    # made feasible, so no schedule needs more stages than there are compressors.
    stage_limit = int(counts.sum()) + 2

    for c in range(1, ncols):
        s = 0
        while True:
            if s > stage_limit:
                raise ValueError(
                    f"column {c} still has {int(left[0][c])} 3:2 and {int(left[1][c])} 2:2 "
                    f"compressors that never become feasible; counts are not schedulable"
                )
            if s >= len(stage32):
                stage32.append(np.zeros(ncols, dtype=np.int64))
                stage22.append(np.zeros(ncols, dtype=np.int64))
                avail.append(np.zeros(ncols, dtype=np.int64))
            if s > 0:
                # leftovers from the stage below plus carries out of column c-1
                avail[s][c] = avail[s - 1][c] + stage32[s - 1][c - 1] + stage22[s - 1][c - 1]
            room = avail[s][c]
            want32, want22 = left[0][c], left[1][c]
            if want32 * 3 + want22 * 2 <= room:
                stage32[s][c], stage22[s][c] = want32, want22
                avail[s][c] = room - 2 * want32 - want22
                left[0][c] = left[1][c] = 0
                break
            fit32 = room // 3
            if want32 >= fit32:
                use32 = fit32
                use22 = 1 if (room % 3 == 2 and want22 >= 1) else 0
            else:
                use32 = want32
                use22 = min(want22, (room - use32 * 3) // 2)
            stage32[s][c], stage22[s][c] = use32, use22
            avail[s][c] = room - 2 * use32 - use22
            left[0][c] -= use32
            left[1][c] -= use22
            s += 1

    return np.stack([np.stack(stage32), np.stack(stage22)])


def num_stages(counts: np.ndarray, width: int) -> int:
    """Number of reduction stages the greedy assignment needs."""
    return assign_stages(counts, width).shape[1]


def to_arch(stages: np.ndarray, width: int) -> list[tuple[int, bool]]:
    """Flatten a stage assignment into the order compressors are wired up.

    Stage-major, descending column within a stage, 3:2 before 2:2 -- the
    order decides which dots each compressor consumes.
    """
    stage32, stage22 = stages[0], stages[1]
    arch: list[tuple[int, bool]] = []
    for s in range(stage32.shape[0]):
        for c in range(n_columns(width) - 1, -1, -1):
            arch.extend([(c, True)] * int(stage32[s][c]))
            arch.extend([(c, False)] * int(stage22[s][c]))
    return arch


# Starting trees: a Wallace-style reduction and an ILP-optimised one, defined
# at 8 and 16 bits. All four are legal under `is_legal`.
INITIAL_STATES: dict[tuple[int, str], list[list[int]]] = {
    (8, "wallace"): [[0, 0, 1, 2, 2, 4, 4, 6, 5, 5, 4, 2, 2, 1, 0],
                     [0, 1, 1, 1, 3, 1, 2, 0, 1, 0, 0, 2, 1, 1, 1]],
    (8, "ilp"): [[0, 0, 1, 2, 3, 4, 4, 6, 5, 5, 4, 3, 2, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]],
    (16, "wallace"): [
        [0, 0, 1, 2, 3, 4, 5, 6, 6, 8, 9, 9, 11, 11, 12, 13,
         13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    (16, "ilp"): [
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 12, 13, 13,
         13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
}


def initial_state(width: int, kind: str = "wallace") -> np.ndarray:
    """Look up a starting tree by name.

    ``wallace`` and ``ilp`` are the only kinds defined, at widths 8 and 16 --
    which is why this project covers those widths and no others.
    """
    try:
        return np.array(INITIAL_STATES[(width, kind)], dtype=np.int64)
    except KeyError:
        known = sorted(INITIAL_STATES)
        raise ValueError(
            f"no {kind!r} starting tree for width {width}. Defined trees are "
            + ", ".join(f"{k} at width {w}" for w, k in known)
        ) from None
