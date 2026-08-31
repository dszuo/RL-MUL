"""Training entry point.

Runs several environments against one agent and one shared PPA cache.  The
environments step in lockstep: actions for all of them come from a single
batched forward pass, then their synthesis jobs run concurrently.  That matters
because synthesis is ~98% of a step's wall time, so the only way to keep a GPU
busy is to have many designs in flight at once.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import random
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from dataclasses import fields as dc_fields

from . import ct
from .agent.dqn import DQNAgent, DQNConfig
from .backend.cache import PPACache
from .backend.rtl import emit_verilog
from .backend.synth import Synthesizer, Toolchain
from .env import CompressorTreeEnv, EnvConfig

log = logging.getLogger("rlmul")

DEFAULTS = {
    "width": 8,
    "env": {"max_stages": 16, "episode_length": 250, "initial_trees": ["wallace", "ilp"],
            "area_weight": 1.0, "delay_weight": 1.0},
    "agent": {},
    "synthesis": {"liberty": "${RLMUL_LIBERTY}", "target_delays": [450, 500, 550, 600],
                  "workdir": None, "timeout_s": 300, "cache": "ppa_cache.sqlite",
                  "yosys": "yosys", "sta": "sta", "driving_cell": "BUF_X1",
                  "output_load": 10.0, "retries": 1},
    "train": {"total_steps": 10_000, "num_envs": 8, "device": "auto", "seed": 0,
              "out_dir": "runs/default", "log_every": 20, "checkpoint_every": 1000,
              "max_seconds": 0},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


SECTION_KEYS = {
    "env": {f.name for f in dc_fields(EnvConfig)} - {"width"},
    "agent": {f.name for f in dc_fields(DQNConfig)},
    "synthesis": set(DEFAULTS["synthesis"]),
    "train": set(DEFAULTS["train"]),
}


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _non_negative_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _delay_list(value) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and all(_positive_number(v) for v in value))


def _non_negative_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _unit_interval(value) -> bool:
    return _number(value) and 0.0 <= value <= 1.0


def _name_list(value) -> bool:
    # `--set env.initial_trees=[wallace]` is not JSON, so it arrives as the
    # string "[wallace]" and silently indexes to "[". Requiring a real list of
    # strings turns that into an error naming the key.
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and all(isinstance(v, str) and v for v in value))


def _size_list(value) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and all(_positive_int(v) for v in value))


# Values that must hold for a run to mean anything. The three marked below are
# the dangerous ones: they pass a name-only check, raise nothing, and produce a
# run that reports success while having measured nothing.
VALUE_RULES: list[tuple[str, str, object, str]] = [
    ("", "width", _positive_int, "a positive integer"),
    ("env", "max_stages", _positive_int, "a positive integer"),
    ("env", "episode_length", _positive_int, "a positive integer"),  # 0 = never walks
    ("agent", "batch_size", _positive_int, "a positive integer"),
    ("agent", "buffer_size", _positive_int, "a positive integer"),
    ("agent", "train_frequency", _positive_int, "a positive integer"),
    ("agent", "target_update", _positive_int, "a positive integer"),
    ("agent", "learning_starts", _non_negative_int, "a non-negative integer"),
    ("agent", "lr", _positive_number, "a positive number"),
    ("synthesis", "target_delays", _delay_list, "a non-empty list of positive numbers"),  # [] = NaN
    ("synthesis", "timeout_s", _positive_number, "a positive number"),
    ("synthesis", "retries", _non_negative_int, "a non-negative integer"),  # -1 = runs no tools
    ("synthesis", "output_load", _positive_number, "a positive number"),
    ("train", "total_steps", _positive_int, "a positive integer"),
    ("train", "num_envs", _positive_int, "a positive integer"),
    ("train", "seed", _non_negative_int, "a non-negative integer"),
    ("train", "log_every", _non_negative_int, "a non-negative integer (0 disables)"),
    ("train", "checkpoint_every", _non_negative_int, "a non-negative integer (0 disables)"),
    ("train", "max_seconds", _non_negative_number, "a non-negative number (0 disables)"),
    ("train", "device", lambda v: isinstance(v, str) and v, "a device string"),
    ("train", "out_dir", lambda v: isinstance(v, str) and v, "a directory path"),
    ("agent", "gamma", _unit_interval, "a number in [0, 1]"),
    ("agent", "eps_start", _unit_interval, "a number in [0, 1]"),
    ("agent", "eps_end", _unit_interval, "a number in [0, 1]"),
    ("agent", "eps_decay_steps", _positive_int, "a positive integer"),
    ("agent", "grad_clip", _positive_number, "a positive number"),
    ("agent", "blocks", _size_list, "a non-empty list of positive integers"),
    ("agent", "channels", _size_list, "a non-empty list of positive integers"),
    ("env", "initial_trees", _name_list, "a non-empty list of tree names"),
    ("env", "area_weight", _non_negative_number, "a non-negative number"),
    ("env", "delay_weight", _non_negative_number, "a non-negative number"),
]


def validate_config(config: dict) -> None:
    """Reject unknown keys and unusable values.

    A typo in a YAML file or a ``--set`` override would otherwise leave the
    default in place and quietly change nothing, which is a miserable way to
    lose a training run. Values matter just as much: several of them pass a
    name-only check and then produce a run that finishes, writes a normal
    looking summary, and has measured nothing at all.
    """
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"unknown config section(s): {', '.join(sorted(unknown))}")
    for section, allowed in SECTION_KEYS.items():
        node = config.get(section, {})
        if not isinstance(node, dict):
            raise SystemExit(f"[{section}] must be a mapping of settings, got {node!r}")
        extra = set(node) - allowed
        if extra:
            raise SystemExit(
                f"unknown key(s) in [{section}]: {', '.join(sorted(extra))}; "
                f"known keys are {', '.join(sorted(allowed))}"
            )
    for section, key, ok, expected in VALUE_RULES:
        node = config.get(section, {}) if section else config
        if not isinstance(node, dict):
            raise SystemExit(f"[{section}] must be a mapping, got {node!r}")
        value = node.get(key)
        if value is None:
            continue
        if not ok(value):
            name = f"{section}.{key}" if section else key
            raise SystemExit(f"{name} must be {expected}, got {value!r}")


def load_config(path: str | None, overrides: list[str]) -> dict:
    # Deep copy: the merge below shares nested dicts with DEFAULTS, and the
    # override loop writes in place, so without this a --set would edit the
    # module-level defaults and leak into every later call.
    config = copy.deepcopy(DEFAULTS)
    if path:
        import yaml
        config = deep_merge(config, yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            node[parts[-1]] = json.loads(raw)
        except json.JSONDecodeError:
            node[parts[-1]] = raw
    validate_config(config)
    return config


def raise_thread_limit(wanted: int = 65_536) -> None:
    """Lift the per-user process/thread soft limit if it is in the way.

    Each environment holds a small pool of synthesis workers, so a few dozen
    environments plus whatever else the account is running can hit a 4096-style
    soft limit and fail with "can't start new thread".  The hard limit is
    normally enormous, so raising the soft limit needs no privileges.
    """
    try:
        import resource  # POSIX only; absent on Windows
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    target = min(wanted, hard)
    if soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (target, hard))
            log.info("raised process limit from %s to %s", soft, target)
        except (ValueError, OSError) as exc:
            log.warning("could not raise process limit from %s: %s", soft, exc)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def build(config: dict):
    width = int(config["width"])
    syn_cfg = dict(config["synthesis"])
    liberty = os.path.expandvars(str(syn_cfg["liberty"]))
    if "$" in liberty or not liberty:
        raise SystemExit(
            "set the standard-cell library, e.g. RLMUL_LIBERTY=/path/to/lib.lib "
            "or --set synthesis.liberty=/path/to/lib.lib"
        )
    out_dir = Path(config["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = syn_cfg.get("cache")
    cache = PPACache(out_dir / cache_path if cache_path else None)

    target_delays = tuple(syn_cfg["target_delays"])
    toolchain = Toolchain(
        liberty=liberty,
        yosys=str(syn_cfg.get("yosys", "yosys")),
        sta=str(syn_cfg.get("sta", "sta")),
        driving_cell=str(syn_cfg.get("driving_cell", "BUF_X1")),
        output_load=float(syn_cfg.get("output_load", 10.0)),
        timeout_s=float(syn_cfg["timeout_s"]),
    )

    n_envs = int(config["train"]["num_envs"])
    env_cfg = EnvConfig(width=width, **config["env"])
    envs = []
    for i in range(n_envs):
        synth = Synthesizer(
            width=width, toolchain=toolchain,
            target_delays=target_delays,
            workdir=Path(syn_cfg["workdir"]) if syn_cfg.get("workdir") else None,
            cache=cache,
            # One worker per constraint: the jobs are independent and there is
            # nothing else for extra workers to do.
            max_workers=len(target_delays),
            retries=int(syn_cfg.get("retries", 1)),
        )
        envs.append(CompressorTreeEnv(env_cfg, synth, seed=config["train"]["seed"] + i))
    return envs, cache


def train(config: dict) -> dict:
    tcfg = config["train"]
    width = int(config["width"])
    out_dir = Path(tcfg["out_dir"])
    set_seed(int(tcfg["seed"]))
    raise_thread_limit()

    # Installed before anything slow: measuring the reference PPA takes minutes
    # at width 16, and a SIGTERM arriving during it would otherwise leave nothing
    # behind. Schedulers stop jobs with SIGTERM, whose default action skips
    # `finally` entirely, so turning it into an exception is what lets the normal
    # shutdown path run at all.
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt("received SIGTERM")

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:
        previous_sigterm = None  # not on the main thread; nothing to restore

    envs, cache = build(config)
    device = pick_device(str(tcfg["device"]))
    obs_shape = envs[0].observation().shape
    agent = DQNAgent(obs_shape, ct.n_actions(width), DQNConfig(**config["agent"]),
                     device=device, seed=int(tcfg["seed"]))
    log.info("training width=%d on %s with %d environments, %d actions",
             width, device, len(envs), ct.n_actions(width))

    observations = [env.reset(seed=int(tcfg["seed"]) + i) for i, env in enumerate(envs)]
    masks = [env.action_mask() for env in envs]

    # Seed the best-so-far from the cheapest starting tree. The first one
    # scores exactly 2.0 only because the normalisation is measured from it;
    # every starting tree is a real candidate.
    best_kind = min(envs[0].references, key=lambda k: envs[0].score(*envs[0].references[k]))
    best_area, best_delay = envs[0].references[best_kind]
    best = {
        "score": envs[0].score(best_area, best_delay),
        "counts": ct.initial_state(width, best_kind),
        "area": best_area, "delay": best_delay, "step": 0,
    }
    _save_best(out_dir, width, best)
    history: list[dict] = []
    log_path = out_dir / "train_log.csv"
    log_file = log_path.open("w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=[
        "step", "wall_s", "epsilon", "loss", "reward", "area", "delay",
        "score", "best_score", "n_stages", "n_compressors", "failures",
    ])
    writer.writeheader()

    pool = ThreadPoolExecutor(max_workers=len(envs), thread_name_prefix="env")
    start = time.time()
    total_steps = int(tcfg["total_steps"])
    budget = float(tcfg.get("max_seconds") or 0)
    failures = 0
    step = 0
    interrupted = False
    # 0 turns these off, the same way train.max_seconds does.
    log_every = int(tcfg["log_every"])
    checkpoint_every = int(tcfg["checkpoint_every"])
    try:
        while step < total_steps:
            legal = [np.flatnonzero(m) for m in masks]
            stuck = [i for i, l in enumerate(legal) if l.size == 0]
            if stuck:
                raise RuntimeError(
                    f"environment(s) {stuck} have no legal edit left; the tree cannot "
                    f"be changed under the configured env.max_stages"
                )
            greedy = agent.act_batch(np.stack(observations), np.stack(masks))
            actions = [
                int(agent.rng.choice(legal[i].tolist()))
                if agent.rng.random() < agent.epsilon else int(greedy[i])
                for i in range(len(envs))
            ]

            results = list(pool.map(lambda p: p[0].step(p[1]), zip(envs, actions)))

            losses = []
            for i, (env, action, result) in enumerate(zip(envs, actions, results)):
                next_mask = env.action_mask()
                agent.observe(observations[i], action, result.reward,
                              result.observation, next_mask)
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
                step += 1

                if result.info.get("synthesis_failed"):
                    failures += 1
                else:
                    score = result.info["score"]
                    if score < best["score"]:
                        best = {"score": score, "counts": result.info["counts"],
                                "area": result.info["area"], "delay": result.info["delay"],
                                "step": step}
                        _save_best(out_dir, width, best)

                if result.truncated:
                    observations[i] = env.reset()
                    masks[i] = env.action_mask()
                else:
                    observations[i] = result.observation
                    masks[i] = next_mask

                row = {
                    "step": step, "wall_s": round(time.time() - start, 2),
                    "epsilon": round(agent.epsilon, 4),
                    "loss": round(losses[-1], 6) if losses else "",
                    "reward": round(result.reward, 6),
                    "area": result.info.get("area", ""),
                    "delay": result.info.get("delay", ""),
                    "score": result.info.get("score", ""),
                    "best_score": round(best["score"], 6),
                    "n_stages": result.info.get("n_stages", ""),
                    "n_compressors": result.info.get("n_compressors", ""),
                    "failures": failures,
                }
                writer.writerow(row)
                history.append(row)
            log_file.flush()

            if log_every and step % log_every < len(envs):
                log.info(
                    "step %6d/%d  eps=%.3f  loss=%s  best=%.5f (area=%.1f delay=%.4f)  "
                    "fail=%d  %.2f steps/s",
                    step, total_steps, agent.epsilon,
                    f"{np.mean(losses):.4f}" if losses else "--",
                    best["score"], best["area"] or 0.0, best["delay"] or 0.0,
                    failures, step / max(1e-9, time.time() - start),
                )
                log_file.flush()
            if checkpoint_every and step % checkpoint_every < len(envs):
                _save_atomically(out_dir / "checkpoint.pt",
                                 lambda p: torch.save(agent.state_dict(), p))
            if budget and time.time() - start > budget:
                log.info("stopping: hit the %.0fs budget", budget)
                break
    except KeyboardInterrupt:
        log.warning("interrupted at step %d; writing out what we have", step)
        interrupted = True
    except Exception:
        # Anything else -- a CUDA OOM, a tool writing something unparseable, a
        # bug -- still gets the checkpoint and the summary written before it
        # propagates. Losing hours of synthesis to an unhandled exception is a
        # worse outcome than the exception itself.
        log.exception("failed at step %d; writing out what we have", step)
        raise
    finally:
        pool.shutdown(wait=True)
        for env in envs:
            close = getattr(env.evaluator, "close", None)
            if close:
                close()
        log_file.close()
        cache.close()
        _save_atomically(out_dir / "checkpoint.pt",
                         lambda p: torch.save(agent.state_dict(), p))
        _write_summary(out_dir, config, envs, best, step, failures, start,
                       interrupted=interrupted)
        # Restored last: everything above can take minutes, and until it is done
        # a second SIGTERM must still reach the handler rather than the default
        # action, which would kill the process outright.
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)

    return json.loads((out_dir / "summary.json").read_text())


def _write_summary(out_dir: Path, config: dict, envs, best: dict,
                   step: int, failures: int, start: float,
                   interrupted: bool = False) -> dict:
    width = int(config["width"])
    elapsed = time.time() - start
    summary = {
        "width": width, "steps": step, "interrupted": interrupted,
        "wall_seconds": round(elapsed, 1),
        "steps_per_second": round(step / max(1e-9, elapsed), 3),
        "synthesis_failures": failures,
        "reference_area": envs[0].ref_area, "reference_delay": envs[0].ref_delay,
        "best_score": best["score"], "best_area": best["area"],
        "best_delay": best["delay"], "best_step": best["step"],
        "best_compressors": int(best["counts"].sum()) if best["counts"] is not None else None,
        "best_stages": ct.num_stages(best["counts"], width) if best["counts"] is not None else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("done: %s", json.dumps(summary))
    return summary


def _save_atomically(path: Path, write) -> None:
    """Write via a temporary file so an interrupted save cannot destroy the old one.

    A ResNet-18 checkpoint is ~90 MB and is rewritten throughout the run; a kill
    landing inside torch.save leaves a truncated file and the previous good copy
    already gone.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _save_best(out_dir: Path, width: int, best: dict) -> None:
    np.savetxt(out_dir / "best_counts.txt", best["counts"], fmt="%d")
    (out_dir / "best_MUL.v").write_text(emit_verilog(best["counts"], width))
    (out_dir / "best.json").write_text(json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best.items()},
        indent=2) + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent to design compressor trees")
    parser.add_argument("-c", "--config", help="YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config key, e.g. --set train.num_envs=16")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(args.config, args.overrides)
    summary = train(config)
    # A run that was stopped part-way wrote out everything it had, but it did
    # not finish: a scheduler must not read that as success.
    return 130 if summary.get("interrupted") else 0


if __name__ == "__main__":
    sys.exit(main())
