# InvertedPendulum-v4: REINFORCE, REINFORCE+Baseline, and A3C

This project extends the Gymnasium REINFORCE tutorial
(https://gymnasium.farama.org/tutorials/training_agents/mujoco_reinforce/)
with two additional policy-gradient variants, trained and compared on
the same environment (`InvertedPendulum-v4`), the same 5 Fibonacci
seeds (`1, 2, 3, 5, 8`), and the same episode budget (`5000` episodes).

## Contents

```
project/
├── reinforce/
│   └── mujoco_reinforce.py            # base algorithm from the tutorial (unmodified logic)
├── reinforce_baseline/
│   └── mujoco_reinforce_baseline.py   # REINFORCE + learned V(s) baseline
├── a3c/
│   └── mujoco_a3c.py                  # Asynchronous Advantage Actor-Critic
├── plot_results.py                    # combines saved .npy results into report plots
├── requirements.txt
└── README.md
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

MuJoCo environments require the `mujoco` python package and, on some
platforms, a working OpenGL/EGL setup for rendering (rendering is not
needed here since we don't call `env.render()`).

## 2. Train each algorithm

Each script trains across all 5 seeds sequentially and saves the
per-seed reward-vs-episode curves as `.npy` files inside `results/`
(one array of shape `(episodes,)` per seed, plus a combined
`(num_seeds, episodes)` array).

```bash
cd reinforce           && python mujoco_reinforce.py          --out_dir ../results
cd ../reinforce_baseline && python mujoco_reinforce_baseline.py --out_dir ../results
cd ../a3c               && python mujoco_a3c.py                --out_dir ../results
```

Optional flags (same across all three scripts):
```
--episodes N        # default 5000
--seeds 1 2 3 5 8    # default Fibonacci seeds
--out_dir DIR        # default ../results
```
`mujoco_a3c.py` additionally accepts `--workers N` (default 4) to set
the number of asynchronous actor-learner processes per seed.

## 3. Generate plots

```bash
cd ..
python plot_results.py --results_dir results --out_dir plots
```

This produces:
- `plots/reinforce.png`, `plots/reinforce_baseline.png`, `plots/a3c.png`
  — one reward-vs-episode plot per algorithm (mean +/- variance band
  across the 5 seeds, same style as the tutorial's plot).
- `plots/comparison_overlay.png` — all three algorithms' (smoothed)
  mean learning curves overlaid on one axis, for the written report.

## Notes on each algorithm

**REINFORCE** (`reinforce/`): exactly the tutorial's implementation —
full-episode Monte-Carlo return `G_t`, single policy network (mean +
std of a Gaussian), one gradient step per episode.

**REINFORCE + Baseline** (`reinforce_baseline/`): adds a second,
independently-trained value network `V(s)`. The advantage
`A_t = G_t - V(s_t)` replaces `G_t` in the policy gradient, which
reduces the variance of the gradient estimate without introducing
bias (since `E[G_t | s_t] - V(s_t) -> 0` as `V` converges to the true
value function). `V` is trained by regression toward `G_t`.

**A3C** (`a3c/`): a shared global actor-critic network is updated
asynchronously by `N_WORKERS` (default 4) parallel worker processes,
each running its own copy of the environment. Workers periodically
sync their local weights from the global network, roll out one
episode, compute an advantage actor-critic loss (policy loss +
value loss - entropy bonus), and push gradients back to the shared
network using Hogwild!-style lock-free asynchronous SGD (`SharedAdam`),
following Mnih et al. (2016). Because training is asynchronous,
exact reproducibility across runs is weaker than for the two
single-threaded algorithms above — this is an inherent property of
A3C, not a bug, and is discussed in the report.
