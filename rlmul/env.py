"""The compressor-tree design environment.

One episode is a walk through legal compressor trees: the agent repeatedly
edits a column, the tree is synthesised, and the reward is how much the
area-delay score improved over the previous step.  Rewarding the *improvement*
rather than the absolute score keeps the signal centred as the walk moves
through very different regions of the design space.

Scores are normalised by the PPA of the starting tree, measured once when the
environment is built, so the reward scale is correct for whatever library and
tool versions are configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import ct

log = logging.getLogger(__name__)


class Evaluator(Protocol):
    """Anything that can price a compressor tree. See `backend.synth.Synthesizer`."""

    def evaluate(self, counts: np.ndarray): ...


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    truncated: bool
    info: dict


@dataclass
class EnvConfig:
    width: int = 8
    max_stages: int = 16
    episode_length: int = 250
    initial_trees: tuple[str, ...] = ("wallace", "ilp")
    area_weight: float = 1.0
    delay_weight: float = 1.0


class CompressorTreeEnv:
    """A single-agent environment over legal compressor trees."""

    def __init__(self, config: EnvConfig, evaluator: Evaluator, seed: int | None = None):
        self.config = config
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed)
        self.width = config.width
        self.n_actions = ct.n_actions(config.width)

        self._counts = ct.initial_state(config.width, config.initial_trees[0])
        self._steps = 0
        self._score = None

        # Reference PPA: the score of every starting tree is measured once, and
        # the first one anchors the normalisation.
        self._reference: dict[str, tuple[float, float]] = {}
        for kind in config.initial_trees:
            tree = ct.initial_state(config.width, kind)
            depth = ct.num_stages(tree, config.width)
            if depth > config.max_stages:
                # observation() pads to max_stages and would silently drop the
                # deeper rows, leaving the agent with an incomplete view of a
                # tree it is nonetheless being scored on.
                raise ValueError(
                    f"the {kind!r} starting tree for width {config.width} needs "
                    f"{depth} reduction stages, more than env.max_stages="
                    f"{config.max_stages}; raise max_stages"
                )
            ppa = evaluator.evaluate(tree)
            if ppa is None:
                raise RuntimeError(
                    f"could not synthesise the {kind!r} starting tree for width "
                    f"{config.width}; check the toolchain before training"
                )
            self._reference[kind] = (ppa.area, ppa.delay)
        self.ref_area, self.ref_delay = self._reference[config.initial_trees[0]]
        log.info(
            "width %d reference PPA: area=%.2f delay=%.4f (%s tree)",
            config.width, self.ref_area, self.ref_delay, config.initial_trees[0],
        )

    # -- gym-ish API -------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        kind = str(self.rng.choice(self.config.initial_trees))
        self._counts = ct.initial_state(self.width, kind)
        self._steps = 0
        area, delay = self._reference[kind]
        self._score = self.score(area, delay)
        return self.observation()

    def step(self, action: int) -> StepResult:
        candidate = ct.apply_action(self._counts, self.width, int(action))
        self._steps += 1
        truncated = self._steps >= self.config.episode_length

        ppa = self.evaluator.evaluate(candidate)
        if ppa is None:
            # Synthesis only fails for reasons outside the design -- a tool that
            # timed out, a full disk, a fork that could not be started on a busy
            # machine. An illegal tree raises instead of landing here. So the
            # edit is simply not taken: the tree is unchanged, and the reward is
            # the zero that follows from the score not having moved.
            log.debug("synthesis failed, staying put at step %d", self._steps)
            return StepResult(
                self.observation(), 0.0, truncated,
                {"synthesis_failed": True},
            )

        self._counts = candidate
        score = self.score(ppa.area, ppa.delay)
        reward = self._score - score
        self._score = score
        return StepResult(
            self.observation(), float(reward), truncated,
            {
                "area": ppa.area,
                "delay": ppa.delay,
                "score": score,
                "n_stages": ct.num_stages(self._counts, self.width),
                "n_compressors": int(self._counts.sum()),
                "counts": self._counts.copy(),
                "synthesis_failed": False,
            },
        )

    # -- state -------------------------------------------------------------

    @property
    def references(self) -> dict[str, tuple[float, float]]:
        """Measured ``(area, delay)`` of each configured starting tree."""
        return dict(self._reference)

    def score(self, area: float, delay: float) -> float:
        """Weighted area-delay score, 1.0 + 1.0 for the reference tree."""
        c = self.config
        return c.area_weight * area / self.ref_area + c.delay_weight * delay / self.ref_delay

    def observation(self) -> np.ndarray:
        """Per-stage compressor counts, shape ``(2, max_stages, 2W-1)``.

        Two planes (3:2 and 2:2), one row per reduction stage, padded with zeros
        so the tensor keeps a fixed shape as the tree's depth changes.
        """
        stages = ct.assign_stages(self._counts, self.width)
        n = stages.shape[1]
        obs = np.zeros((2, self.config.max_stages, ct.n_columns(self.width)), dtype=np.float32)
        keep = min(n, self.config.max_stages)
        obs[:, :keep] = stages[:, :keep]
        return obs

    def action_mask(self) -> np.ndarray:
        return ct.legal_action_mask(self._counts, self.width, self.config.max_stages)
