# ViProX: An Implicit Visual Proprioception Expert System

Official code release for the ESWA paper *ViProX* (Ms. Ref. No. ESWA-D-26-29997).

ViProX endows an imitation-learning policy with **implicit proprioception**: the
policy observes only camera images (no joint encoders at inference), while an
auxiliary regression head is trained to recover joint positions/velocities
(`qpos`, `qvel`) from a short visual history, forcing the visual representation
to encode the kinematic state. The codebase is derived from
[ACT](https://github.com/tonyzhaozh/act) (Tony Z. Zhao et al., Stanford) and
uses the ALOHA simulation environments; the transformer backbone is derived
from [DETR](https://github.com/facebookresearch/detr).

## Repository layout

```
viprox/
  imitate_episodes.py        # training / evaluation entry point
  record_sim_episodes.py     # record scripted-policy demonstrations
  record_sim_episodes_multi.py
  policy.py                  # ACT / CNN-MLP / Diffusion / FlowMatching policies
  utils.py                   # data loading, photometric augmentation, visual history
  constants.py               # task configs, normalization constants
  sim_env.py, ee_sim_env.py  # MuJoCo simulation environments
  scripted_policy.py         # expert policies (figure-8, insertion, transfer)
  detr/                      # transformer backbone (forked from DETR/ACT)
  assets/                    # MuJoCo XML models and meshes
```

## Installation

```bash
conda create -n viprox python=3.9 -y
conda activate viprox
pip install torch torchvision          # pick the build matching your CUDA
pip install -r requirements.txt
```

Optional dependencies:

- **robomimic 0.3.0** — only needed for the `DiffusionPolicy` and
  `FlowMatchingPolicy` baselines:
  `pip install robomimic==0.3.0` (or `pip install -e` from the
  [source repo](https://github.com/ARISE-Initiative/robomimic)).
- **RAFT** — only needed for the RAFT optical-flow baseline
  (`--dynamics_mode raft`). Clone the official repo into this directory and
  download its weights:
  ```bash
  git clone https://github.com/princeton-vl/RAFT.git
  cd RAFT && ./download_models.sh && cd ..
  ```
  RAFT is loaded lazily from `./RAFT/core`; if it is missing, only the RAFT
  baseline is unavailable and everything else still runs.

Datasets are looked up under `./data` by default; override with the
`VIPROX_DATA_DIR` environment variable (see `constants.py`).

## Quick start (simulation)

Two simulated tasks are included, plus the original ACT `sim_transfer_cube_scripted`:

| Task | Episodes | Length |
|---|---|---|
| `sim_figure8_scripted` | 50 | 800 |
| `sim_insertion_scripted` | 50 | 400 |
| `sim_transfer_cube_scripted` | 50 | 400 |

### 1. Record demonstrations with the scripted expert

```bash
python record_sim_episodes.py \
    --task_name sim_figure8_scripted \
    --dataset_dir ./data/sim_figure8_scripted \
    --num_episodes 50
```

### 2. Train ViProX (ACT backbone + visual history + proprioception aux loss)

```bash
python imitate_episodes.py \
    --task_name sim_figure8_scripted \
    --ckpt_dir ./ckpt/figure8_viprox \
    --policy_class ACT --kl_weight 10 \
    --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 16 --lr 1e-5 --num_steps 20000 --seed 0 \
    --proprio_weight 10 --qvel_weight 0.5 \
    --visual_history_len 2 \
    --photo_aug --frame_drop_prob 0.1 \
    --eval_every 100000000 --validate_every 500 --save_every 5000
```

Baselines:

- **ACT** (standard, single-frame, joint encoders on): drop
  `--visual_history_len` (defaults differ per ablation) and set
  `--proprio_weight 0`; disable the visual-history encoder path per the
  ablation configs in the paper.
- **Diffusion Policy**: `--policy_class Diffusion` (requires robomimic).
- **RAFT flow baseline**: `--dynamics_mode raft` (requires the RAFT clone above;
  the flow baseline uses exactly `--visual_history_len 2`).

### 3. Evaluate

```bash
python imitate_episodes.py \
    --task_name sim_figure8_scripted \
    --ckpt_dir ./ckpt/figure8_viprox \
    --policy_class ACT --kl_weight 10 \
    --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 \
    --eval --visual_history_len 2 --eval_frame_stride 1 \
    --temporal_agg
```

Evaluation renders on-screen by default; for headless servers set
`MUJOCO_GL=egl` (GPU) or run on CPU. The default rollout count is 50; for a
quick check set `VIPROX_EVAL_NUM_ROLLOUTS=1`.

## Loss-weight convention (paper vs. code)

The code optimizes

```
L = L1(action) + kl_weight * KL + proprio_weight * ( MSE(qpos) + qvel_weight * MSE(qvel) )
```

Paper Eq. (2) writes the auxiliary term as `λ1 * MSE(qpos) + λ2 * MSE(qvel)`
with `λ1 = 1.0, λ2 = 0.5`, which corresponds to the inner parenthesis
(`qvel_weight = λ2 / λ1 = 0.5`). `proprio_weight = 10` is the global weight
`λ_aux` of the whole auxiliary term in the paper's Eq. (4), and
`kl_weight = 10` is `β`. The paper states both: `λ1, λ2` fix the shape of the
auxiliary term and `λ_aux` its overall strength against the reconstruction
term.

## Photometric augmentation / history window

`utils.py` implements the training-time photometric augmentation
(brightness / contrast / saturation / gamma, sampled per sample and shared
across the history frames and cameras of that sample) and random frame-drop.
The evaluation-time corruption harness used in the paper's robustness study
is in `imitate_episodes.py`: photometric shifts via `--eval_photo_perturb
{mild,severe}`, temporal staleness via `--eval_frame_drop` /
`--eval_frame_stride`, and the scripted proprioceptive corruptions applied to
qpos-consuming baselines via `--eval_qpos_noise_std`, `--eval_qpos_drift`
and `--eval_qpos_dropout`.

## Checkpoints

Pretrained checkpoints are not bundled with this repository. Every reported
result can be regenerated end-to-end with the commands above (dataset
recording, training, and evaluation), using the seeds given in the paper.

## Citation

```bibtex
@article{viprox2026,
  title  = {ViProX: An Implicit Visual Proprioception Expert System},
  author = {...},
  journal= {Expert Systems with Applications},
  year   = {2026},
  note   = {under revision, Ms. Ref. No. ESWA-D-26-29997}
}
```

## License and attribution

- ViProX modifications are released under the MIT License (see `LICENSE`).
- `detr/` is derived from [DETR](https://github.com/facebookresearch/detr)
  and retains its original Apache 2.0 license (see `detr/LICENSE`).
- The simulation environments and training loop are derived from
  [ACT](https://github.com/tonyzhaozh/act) (MIT license), Copyright (c) 2023
  Tony Z. Zhao.
- The RAFT baseline wraps [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)
  (BSD 3-Clause), which is not vendored here.
