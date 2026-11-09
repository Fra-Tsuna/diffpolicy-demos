# Diffusion Policy on PushT and ALOHA

LeRobot's Diffusion Policy trained with Accelerate, Hydra and W&B:

| Experiment | Dataset | Simulator | Observations | Actions |
| --- | --- | --- | --- | --- |
| PushT (default) | [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht), 206 episodes at 10 Hz | `PushT-v0` | 96x96 image, 2-D agent position | 2-D target position |
| `experiment=aloha_transfer_cube` | [`lerobot/aloha_sim_transfer_cube_human`](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human), 50 episodes at 50 Hz | `AlohaTransferCube-v0` | 480x640 top camera, 14-D joint positions | 14-D joint targets |

## Layout

| Path | Contents |
| --- | --- |
| `main.py` | Training. |
| `eval.py` | Offline evaluation (denoising loss) on the test or val split. |
| `rollout.py` | Closed-loop rollouts in the simulator, with MP4 videos and success metrics. |
| `src/` | Dataset, model wrapper, trainer (EMA, checkpointing, resume) and inference loop. |
| `config/` | Hydra configs; `experiment/` holds the presets and `env/` the simulators. |
| `scripts/` | Shell wrappers, see [scripts/README.md](scripts/README.md). |

## Setup

Requires conda and an NVIDIA driver >= 525 (torch comes from the CUDA 12.6 wheels).

```bash
scripts/setup.sh    # creates the `diffpolicy` conda env
wandb login         # or pass wandb.mode=offline
```

## Usage

```bash
scripts/train.sh                                    # PushT
scripts/train_aloha.sh train_dataloader.num_workers=8

scripts/eval.sh <run_dir>/checkpoints/best.pth experiment=aloha_transfer_cube [apply_ema=true]
scripts/rollout.sh <run_dir>/checkpoints/best.pth experiment=aloha_transfer_cube n_episodes=50 [apply_ema=true]

scripts/test.sh
```

Drop `experiment=aloha_transfer_cube` for PushT. Extra arguments are Hydra overrides. `NUM_GPUS=<n>` trains on several GPUs; `resume_from=<run_dir>/checkpoints/last` resumes. `apply_ema=true` needs a checkpoint trained with `trainer.use_ema=true`.

`best.pth` is selected by `trainer.best_metric`: `loss` (default, lowest validation loss) or `success_rate` (ALOHA), from rollouts run at each validation on fixed seeds (`trainer.rollout.seed + i`).

### Pretrained policy

`lerobot/diffusion_pusht` is in the pre-0.4 LeRobot format; convert it once into `ckpt/diffusion_pusht` (the rollout default):

```bash
scripts/download_pretrained.sh
scripts/rollout.sh n_episodes=10
```

There is no pretrained ALOHA policy, so ALOHA rollouts need `checkpoint_path`.

## ALOHA image preprocessing

The default center crop removed the useful information from ALOHA frames: at the start of an episode the grippers sit at the left and right image borders, so cropping cut them out. ALOHA therefore disables the crop (`model.crop_ratio=1.0`) and instead resizes the whole 480x640 frame to 192x256 (`model.resize_shape`), keeping the aspect ratio. 192x256 rather than 96x128, because at 96x128 the cube was only about 4x4 pixels. Preprocessing runs inside the policy and is saved with the checkpoint, so training, evaluation and rollouts match.

## Rollouts

Episode `i` uses seed `seed + i` for the initial state and the diffusion noise, so checkpoints are compared on the same episodes. Videos overlay the denoising of each action chunk and the executed plan; for ALOHA the 14 joint targets are mapped to gripper fingertip trajectories by forward kinematics. ALOHA videos use the `angle` scene camera (`video_camera`), while the policy sees `top`. `diffusion_overlay=false` records plain videos.

## Outputs and splits

| Command | Directory |
| --- | --- |
| train | `outputs/<name>/<date>/<time>/` (`checkpoints/best.pth`, `checkpoints/last/`, `rollouts/iter_<step>/`) |
| eval | `outputs/<name>/eval/<date>/<time>/metrics.json` |
| rollout | `outputs/<name>/rollout/<date>/<time>/videos/` (MP4s, `metrics.json`) |

Episodes are split by whole episode, 80/10/10 (40/5/5 for ALOHA). `eval.py` uses the test split; `split=val` selects validation. Checkpoints store the class path `src.models.model.DiffusionPolicyWrapper`, so renaming it breaks old checkpoints.
