"""Synthesis and static timing for a generated multiplier.

Area comes from ``yosys stat -liberty``, which sums the standard-cell areas of
the mapped netlist; delay comes from OpenSTA's critical-path arrival time.  A
design is synthesised once per target delay so the reward reflects a spread of
timing constraints rather than a single operating point.

Every external command runs with a timeout and a return-code check.  A tool that
crashes, hangs, or writes an unparseable report yields ``None`` for that job:
synthesis failures are a normal part of exploring a large design space and must
not take the training run down with them.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import signal
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import ct
from .cache import PPACache, tree_key
from .rtl import emit_verilog

log = logging.getLogger(__name__)

# `stat` prints a per-module line and, when there is hierarchy, a final
# "top module" line carrying the total. Matching the per-module line first would
# report the top module's own cells while excluding everything it instantiates.
_AREA_RE = re.compile(r"Chip area for top module '\\?(\w+)':\s*([0-9.]+)")
_AREA_FLAT_RE = re.compile(r"Chip area for module '\\?(\w+)':\s*([0-9.]+)")
_CELL_RE = re.compile(r"^\s*cell\s*\(\s*\"?([A-Za-z_][\w$]*)\"?\s*\)", re.MULTILINE)
_ARRIVAL_RE = re.compile(r"^ARRIVAL\s+([0-9.eE+-]+)\s*$", re.MULTILINE)

# How long to wait for a killed tool to be reaped before giving up on it.
_REAP_GRACE_S = 10.0

# The library is linked into each job directory under this name, so no
# user-controlled path is ever interpolated into a tool script.
_LIBERTY_LINK = "library.lib"


@dataclass(frozen=True)
class PPA:
    """Area in square microns, delay in nanoseconds."""
    area: float
    delay: float


@dataclass
class Toolchain:
    """Where the tools and the standard-cell library live."""
    liberty: Path
    yosys: str = "yosys"
    sta: str = "sta"
    driving_cell: str = "BUF_X1"
    output_load: float = 10.0
    timeout_s: float = 300.0
    _checked: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.liberty = Path(self.liberty).expanduser().resolve()
        self.output_load = float(self.output_load)
        self.timeout_s = float(self.timeout_s)

    def cache_tag(self) -> str:
        """Identity of everything that changes a synthesis result.

        A cached number is only reusable under an identical setup. Keying on the
        Liberty file's *name* is not enough -- ``stdcells.lib`` and
        ``typical.lib`` are everywhere, so two different PDKs collide and one
        library's area is served for another with no warning. The drive
        constraints matter too: they reach ABC and change the netlist.
        """
        parts = "|".join((
            str(self.liberty), self.driving_cell, f"{self.output_load:g}",
            # The tools are configurable, so a yosys upgrade or a different
            # build must not silently reuse the previous one's numbers.
            shutil.which(self.yosys) or self.yosys,
            shutil.which(self.sta) or self.sta,
        ))
        return hashlib.sha1(parts.encode()).hexdigest()[:16]

    def check(self) -> None:
        """Fail early and legibly when the environment is not set up.

        Every environment shares one toolchain, so this runs once: scanning a
        Liberty file is cheap but not free, and commercial libraries run to
        hundreds of megabytes.
        """
        if self._checked:
            return
        missing = [exe for exe in (self.yosys, self.sta) if shutil.which(exe) is None]
        if missing:
            raise FileNotFoundError(
                f"tool(s) not on PATH: {', '.join(missing)}. "
                "For OpenROAD-flow-scripts run: source <ORFS>/setup_env.sh"
            )
        if not self.liberty.is_file():
            raise FileNotFoundError(f"liberty file not found: {self.liberty}")
        self._check_driving_cell()
        self._checked = True

    def _check_driving_cell(self) -> None:
        """Refuse a driving cell the library does not define.

        ABC reports an unknown driving cell on stdout and yosys still exits 0,
        so the run continues with the input drive constraint quietly dropped and
        reports PPA that is wrong by several percent -- the worst kind of
        failure, since the numbers look entirely plausible. Swapping libraries is
        the first thing anyone does with this project, so it is worth one pass
        over the Liberty file at startup.
        """
        cells = set(_CELL_RE.findall(self.liberty.read_text(errors="ignore")))
        if not cells:
            log.warning("found no cell definitions in %s; skipping the driving-cell check",
                        self.liberty)
            return
        if self.driving_cell in cells:
            return
        buffers = sorted(c for c in cells if "buf" in c.lower())[:8]
        hint = f" Buffers in this library include: {', '.join(buffers)}." if buffers else ""
        raise ValueError(
            f"driving cell {self.driving_cell!r} is not defined in {self.liberty.name}. "
            f"ABC would silently drop the input drive constraint and report PPA that "
            f"looks plausible but is wrong.{hint}"
        )


@dataclass
class Synthesizer:
    """Turn a compressor tree into averaged PPA across several target delays."""

    width: int
    toolchain: Toolchain
    target_delays: tuple[float, ...] = (450.0, 500.0, 550.0, 600.0)
    workdir: Path | None = None
    cache: PPACache | None = None
    max_workers: int = 4
    retries: int = 1
    _tag: str = field(default="", init=False)
    _pool: ThreadPoolExecutor = field(default=None, init=False, repr=False)
    _live: set = field(default_factory=set, init=False, repr=False)
    _live_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.toolchain.check()
        self.target_delays = tuple(float(d) for d in self.target_delays)
        # Results are only comparable within one synthesis setup, so the whole
        # setup goes into the key.
        self._tag = self.toolchain.cache_tag()
        # One pool for the lifetime of the synthesizer. Building a fresh pool per
        # step churns through threads, and with several environments running at
        # once that churn is enough to exhaust the per-user thread limit on a
        # shared machine.
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, min(self.max_workers, len(self.target_delays))),
            thread_name_prefix="synth",
        )

    def close(self) -> None:
        """Stop accepting work and take any running tool down with us.

        ``start_new_session`` puts the tools in their own process group so a
        timeout can reap them, but that also means they no longer die when
        something kills the trainer's group -- a scheduler's SIGTERM would leave
        them running. Killing them here, from the shutdown path the trainer
        already calls, closes that.
        """
        self._closing = True
        with self._live_lock:
            live = list(self._live)
        for proc in live:
            self._kill_tree(proc)
        self._pool.shutdown(wait=True)

    def evaluate(self, counts: np.ndarray) -> PPA | None:
        """Mean area and delay over the target delays, or ``None`` if any job failed."""
        results = self.evaluate_per_constraint(counts)
        if any(r is None for r in results):
            return None
        return PPA(
            area=float(np.mean([r.area for r in results])),
            delay=float(np.mean([r.delay for r in results])),
        )

    def evaluate_per_constraint(self, counts: np.ndarray) -> list[PPA | None]:
        """One :class:`PPA` per target delay, in the order they were configured."""
        counts = np.asarray(counts, dtype=np.int64)
        if not ct.is_legal(counts, self.width):
            raise ValueError("refusing to synthesise an illegal compressor tree")

        pending: list[tuple[int, float, str]] = []
        results: list[PPA | None] = [None] * len(self.target_delays)
        for i, delay in enumerate(self.target_delays):
            key = tree_key(counts, self.width, delay, self._tag)
            hit = self.cache.get(key) if self.cache is not None else None
            if hit is not None:
                results[i] = PPA(area=hit[0], delay=hit[1])
            else:
                pending.append((i, delay, key))
        if not pending:
            return results

        verilog = emit_verilog(counts, self.width)
        futures = {}
        for i, delay, key in pending:
            try:
                futures[self._pool.submit(self._run_one, verilog, delay)] = (i, key)
            except RuntimeError as exc:
                # The per-user thread limit is shared with everything else on the
                # account, so it can be hit mid-run by an unrelated job. Losing
                # one measurement beats losing the run.
                log.warning("could not start a synthesis worker: %s", exc)
        for future, (i, key) in futures.items():
            ppa = future.result()
            results[i] = ppa
            if ppa is not None and self.cache is not None:
                self.cache.put(key, ppa.area, ppa.delay)
        return results

    # -- internals ---------------------------------------------------------

    def _run_one(self, verilog: str, target_delay: float) -> PPA | None:
        try:
            return self._synthesise(verilog, target_delay)
        except OSError as exc:
            # A workdir that is full, read-only, or not a directory must fail
            # this one job, not the training run -- the same contract the tool
            # invocations below already keep.
            log.warning("could not set up a synthesis workdir: %s", exc)
            return None

    def _synthesise(self, verilog: str, target_delay: float) -> PPA | None:
        base = self.workdir
        if base is not None:
            base = Path(base)
            base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base, prefix="syn-") as tmp:
            d = Path(tmp)
            # Reference the library through a fixed name inside the job
            # directory instead of interpolating its path into the scripts.
            # yosys cannot parse a path containing a space with or without
            # quotes, and an unbraced path in Tcl is subject to command
            # substitution -- a config value should never be able to run code.
            link = d / _LIBERTY_LINK
            try:
                link.symlink_to(self.toolchain.liberty)
            except OSError:
                shutil.copy(self.toolchain.liberty, link)
            (d / "MUL.v").write_text(verilog)
            (d / "abc_constr").write_text(
                f"set_driving_cell {self.toolchain.driving_cell}\n"
                f"set_load {self.toolchain.output_load} [all_outputs]\n"
            )
            (d / "syn.ys").write_text(self._yosys_script(target_delay))
            (d / "sta.tcl").write_text(self._sta_script())

            out = self._run([self.toolchain.yosys, "syn.ys"], d)
            if out is None:
                return None
            area = self._parse_area(out, target_delay)
            if area is None:
                return None
            if not (d / "netlist.v").is_file():
                log.warning("yosys wrote no netlist at target delay %g", target_delay)
                return None

            out = self._run([self.toolchain.sta, "sta.tcl"], d)
            if out is None:
                return None
            delay = self._parse_delay(out, target_delay)
            if delay is None:
                return None
            return PPA(area=area, delay=delay)

    def _yosys_script(self, target_delay: float) -> str:
        return (
            "read_verilog MUL.v\n"
            "synth -top MUL\n"
            f"abc -D {target_delay:g} -constr abc_constr -liberty {_LIBERTY_LINK}\n"
            "opt_clean\n"
            "write_verilog -noattr netlist.v\n"
            f"stat -liberty {_LIBERTY_LINK}\n"
        )

    def _sta_script(self) -> str:
        return (
            # Braced so Tcl performs no substitution on it.
            f"read_liberty {{{_LIBERTY_LINK}}}\n"
            "read_verilog netlist.v\n"
            "link_design MUL\n"
            "set_max_delay -from [all_inputs] 0\n"
            "set paths [find_timing_paths -sort_by_slack]\n"
            "if { [llength $paths] == 0 } { puts \"NOPATH\"; exit 0 }\n"
            "puts \"ARRIVAL [sta::format_time [[[lindex $paths 0] path] arrival] 6]\"\n"
            "exit 0\n"
        )

    def _run(self, argv: list[str], cwd: Path) -> str | None:
        """Run a tool, retrying once on a non-zero exit.

        On a busy shared machine these tools fail transiently -- yosys shells
        out to ABC, and when the host is out of process slots that fork fails
        and yosys exits non-zero through no fault of the design.  One retry
        turns most of those back into a usable sample; a timeout is not retried
        because it costs the full budget again.
        """
        last = "no output"
        for attempt in range(self.retries + 1):
            try:
                proc = self._run_once(argv, cwd)
            except subprocess.TimeoutExpired:
                log.warning("%s timed out after %.0fs", argv[0], self.toolchain.timeout_s)
                return None
            except OSError as exc:
                log.warning("could not run %s: %s", argv[0], exc)
                return None
            if proc.returncode == 0:
                return proc.stdout
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            last = tail[-1] if tail else "no output"
            log.debug("%s exited %d (attempt %d): %s",
                      argv[0], proc.returncode, attempt + 1, last)
        log.warning("%s failed after %d attempts: %s", argv[0], self.retries + 1, last)
        return None

    def _run_once(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
        """Run one tool, killing its whole process group if it times out.

        yosys shells out to a separate ``yosys-abc`` binary, and killing only the
        direct child leaves that grandchild running forever -- an invisible leak
        that accumulates over a long run on a shared machine. Putting the child
        in its own session lets the group be signalled as a unit.
        """
        popen_kwargs = {}
        if hasattr(os, "killpg") and hasattr(os, "setsid"):
            popen_kwargs["start_new_session"] = True
        if self._closing:
            return subprocess.CompletedProcess(argv, 1, "", "synthesizer is closing")
        proc = subprocess.Popen(
            argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            # errors="replace": a tool that emits a stray non-UTF-8 byte -- an
            # ABC crash, or simply a banner under a C locale -- would otherwise
            # raise UnicodeDecodeError, which is a ValueError and so slips past
            # every OSError guard on this path and kills the whole run.
            text=True, errors="replace", **popen_kwargs,
        )
        with self._live_lock:
            self._live.add(proc)
        try:
            out, err = proc.communicate(timeout=self.toolchain.timeout_s)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            # Reap with wait(), never communicate(): a grandchild that started
            # its own session escapes the group kill and keeps the inherited
            # pipes open, and communicate() would then block on an EOF that
            # never comes -- a silent hang, which is worse than the leak this
            # is guarding against.
            try:
                proc.wait(timeout=_REAP_GRACE_S)
            except subprocess.TimeoutExpired:
                log.warning("%s survived SIGKILL; abandoning it", argv[0])
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
            raise
        finally:
            with self._live_lock:
                self._live.discard(proc)
        return subprocess.CompletedProcess(argv, proc.returncode, out, err)

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (AttributeError, OSError, ProcessLookupError):
            proc.kill()

    @staticmethod
    def _parse_area(text: str, target_delay: float) -> float | None:
        matches = _AREA_RE.findall(text) or _AREA_FLAT_RE.findall(text)
        totals = [float(area) for name, area in matches if name == "MUL"]
        if not totals:
            log.warning("no area for module MUL at target delay %g", target_delay)
            return None
        return totals[-1]

    @staticmethod
    def _parse_delay(text: str, target_delay: float) -> float | None:
        match = _ARRIVAL_RE.search(text)
        if match is None:
            log.warning("no timing path reported at target delay %g", target_delay)
            return None
        return float(match.group(1))
