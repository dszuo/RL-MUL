# RL-MUL

Reinforcement learning for multiplier design.

A DQN agent edits the compressor tree of an unsigned `W x W` multiplier one
column at a time. Each candidate tree is turned into Verilog, synthesised to a
standard-cell library under several timing constraints, and scored on the
resulting area and delay.

| | |
|---|---|
| State | 3:2 and 2:2 compressor counts per column, `(2, 2W-1)` |
| Action | add or remove a 2:2, or swap a 3:2 and a 2:2, in one column |
| Reward | improvement in the weighted area-delay score over the previous step |
| RTL | pure-Python Verilog writer |
| Synthesis | yosys + ABC for area, OpenSTA for critical-path delay |

## Install

```bash
pip install -e .
```

Training also needs `yosys`, OpenSTA (`sta`) and a Liberty file. Any library
works: point `RLMUL_LIBERTY` at it and set `synthesis.driving_cell` to a buffer
it defines. With [OpenROAD-flow-scripts][orfs], whose `setup_env.sh` puts both
tools on `PATH`:

```bash
source /path/to/OpenROAD-flow-scripts/setup_env.sh
export RLMUL_LIBERTY=.../flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
```

[orfs]: https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts

## Use

```bash
python -m rlmul.train -c configs/mul8.yaml --set train.total_steps=200   # smoke test
python -m rlmul.train -c configs/mul8.yaml                              # real run
python -m rlmul.train -c configs/mul16.yaml --set train.num_envs=16
```

`rlmul-train` is an equivalent console script. The configs document every knob;
`--set key.path=value` overrides one, parsed as JSON, and anything unrecognised
is rejected rather than ignored. Each episode starts from one of
`env.initial_trees`, drawn at random.

A run writes `train_log.csv`, `summary.json`, `checkpoint.pt`, and the best
design as `best_MUL.v` / `best_counts.txt` / `best.json`. Emitting a multiplier
and checking it against every input pair needs no tools at all:

```python
from rlmul import ct
from rlmul.backend import rtl, sim

print(rtl.emit_verilog(ct.initial_state(8), 8))
sim.check_exhaustive(ct.initial_state(8), 8)   # raises on the first wrong product
```

## Layout

`ct.py` is the state space, `env.py` the environment, `train.py` the entry
point. `backend/` emits Verilog (`rtl.py`), simulates it (`sim.py`), synthesises
it (`synth.py`) and memoises the result (`cache.py`); `agent/` is the Q-network
(`net.py`) and Double DQN (`dqn.py`).

## Citation

```bibtex
@inproceedings{rlmul,
  title     = {RL-MUL: Multiplier Design Optimization with Deep Reinforcement Learning},
  author    = {Zuo, Dongsheng and Ouyang, Yikang and Ma, Yuzhe},
  booktitle = {ACM/IEEE Design Automation Conference (DAC)},
  year      = {2023},
  doi       = {10.1109/DAC56929.2023.10247941}
}

@article{rlmul2,
  title   = {RL-MUL 2.0: Multiplier Design Optimization with Parallel Deep
             Reinforcement Learning and Space Reduction},
  author  = {Zuo, Dongsheng and Zhu, Jiadong and Ouyang, Yikang and Ma, Yuzhe},
  journal = {ACM Transactions on Design Automation of Electronic Systems},
  volume  = {31},
  number  = {4},
  pages   = {1--20},
  year    = {2026},
  doi     = {10.1145/3711850}
}
```

## License

MIT -- see [LICENSE](LICENSE).
