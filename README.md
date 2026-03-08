<div align="center">

# SMART Flow

RoadNet + SMART history memory + sparse factorized flow matching decoder for multi-agent motion generation.

</div>

## Overview

This fork keeps the original SMART map/token preprocessing pipeline and replaces the agent next-token prediction head with a sparse factorized conditional flow matching head.

What stays the same:

- `smart/modules/map_decoder.py`: RoadNet-style map encoder
- agent/map token preprocessing
- training and validation entrypoints: `train.py`, `val.py`
- SMART-style sparse factorized context encoding: temporal, map-to-agent, agent-to-agent

What changes:

- `smart/modules/agent_flow_decoder.py`: 2.0 s future generator
- `smart/utils/flow_traj.py`: chunking, OT noising, assembly, midpoint ODE, warm-start helpers
- `smart/model/smart.py`: flow loss, overlap loss, rollout validation

The implemented agent head uses:

- 2.0 s future window
- 4 overlapping 0.5 s segments (`6` points each, overlap `1`)
- 3 s causal token history from SMART factorized context encoding
- current-state anchor token
- joint all-agent training target selection
- open-loop pretraining and optional short closed-loop fine-tuning

## Repository Layout

- [train.py](/home/user/PycharmProjects/SMART/train.py): training entrypoint
- [val.py](/home/user/PycharmProjects/SMART/val.py): validation entrypoint
- [smart/model/smart.py](/home/user/PycharmProjects/SMART/smart/model/smart.py): Lightning module, losses, rollout metrics
- [smart/modules/agent_flow_decoder.py](/home/user/PycharmProjects/SMART/smart/modules/agent_flow_decoder.py): flow decoder
- [smart/utils/flow_traj.py](/home/user/PycharmProjects/SMART/smart/utils/flow_traj.py): flow trajectory helpers
- [configs/train/train_flow.yaml](/home/user/PycharmProjects/SMART/configs/train/train_flow.yaml): open-loop pretraining config
- [configs/train/train_flow_finetune.yaml](/home/user/PycharmProjects/SMART/configs/train/train_flow_finetune.yaml): short closed-loop fine-tuning config
- [configs/validation/validation_flow.yaml](/home/user/PycharmProjects/SMART/configs/validation/validation_flow.yaml): validation config

## Environment

The flow implementation does not add a new external diffusion dependency. It uses the original SMART stack plus local OT noising and midpoint ODE code.

```bash
conda env create -f environment.yml
conda activate SMART
pip install -r requirements.txt
```

If PyG installation fails:

```bash
bash scripts/install_pyg.sh
```

You still need the Waymo Open Dataset API if you train or validate on WOMD/WOSAC-format data.

## Data Preparation

Expected raw data layout:

```text
SMART
├── data
│   ├── waymo
│   │   ├── scenario
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
```

Preprocess raw scenarios:

```bash
python data_preprocess.py \
  --input_dir ./data/waymo/scenario/training \
  --output_dir ./data/waymo_processed/training
```

Do the same for validation/testing if needed.

The sample configs currently point to `data/valid_demo` for quick smoke tests. Before real training, edit the `train_raw_dir` and `val_raw_dir` fields in the config files.

## Training

### 1. Open-loop pretraining

```bash
python train.py \
  --config configs/train/train_flow.yaml \
  --save_ckpt_path ./checkpoints/flow_pretrain
```

What this does:

- reuses SMART map preprocessing and map encoder
- samples up to `anchor_chunk_k=4` anchors per scene
- predicts 4 future segments over a 2.0 s window
- optimizes `flow_loss + overlap_loss`

Checkpoint selection monitors `val_minADE` and keeps the best 5 checkpoints.

### 2. Short closed-loop fine-tuning

After pretraining, run:

```bash
python train.py \
  --config configs/train/train_flow_finetune.yaml \
  --pretrain_ckpt ./checkpoints/flow_pretrain/epoch=XX.ckpt \
  --save_ckpt_path ./checkpoints/flow_finetune
```

What changes in this stage:

- learning rate is reduced
- `closed_loop_steps=4`
- the training loss becomes `flow_loss + overlap_loss + short_rollout_loss`

This keeps the code path the same. There is no second trainer script.

## Validation

```bash
python val.py \
  --config configs/validation/validation_flow.yaml \
  --pretrain_ckpt ./checkpoints/flow_finetune/epoch=YY.ckpt
```

Validation runs:

- open-loop flow loss
- full 8 s closed-loop rollout
- `val_minADE`
- `val_minFDE`

## Minimal Config Surface

The new implementation only introduces four decoder-side runtime controls:

- `future_window_steps: 20`
- `ode_steps: 4`
- `anchor_chunk_k: 4`
- `closed_loop_steps: 0` or `4`

Everything else stays aligned with the original SMART defaults:

- `hidden_dim: 128`
- `num_agent_layers: 6`
- `pl2a_radius: 30`
- `a2a_radius: 60`
- `time_span: 30`
- `shift: 5`

## Important Notes

- The agent NTP training path is no longer used by `train.py` or `val.py`.
- Map token masking and map encoder behavior are unchanged.
- Joint all-agent training is enabled by removing the previous random 32-agent cap in `WaymoTargetBuilder`.
- The implementation intentionally keeps sparse factorized attention instead of importing Flow-Planner's global fusion blocks.

## Practical Run Order

1. Install dependencies.
2. Preprocess Waymo data.
3. Edit the raw/processed data paths inside [configs/train/train_flow.yaml](/home/user/PycharmProjects/SMART/configs/train/train_flow.yaml) and [configs/validation/validation_flow.yaml](/home/user/PycharmProjects/SMART/configs/validation/validation_flow.yaml).
4. Run open-loop pretraining.
5. Run short closed-loop fine-tuning from the best pretrain checkpoint.
6. Run validation on the fine-tuned checkpoint.

## Verification Performed In This Workspace

Static syntax verification passed for the modified code:

```bash
python3 -m py_compile train.py val.py smart/model/smart.py \
  smart/modules/smart_decoder.py smart/modules/agent_flow_decoder.py \
  smart/utils/flow_traj.py smart/transforms/target_builder.py
```

Full training/rollout execution was not run in this workspace because the current Python environment does not have the project dependencies installed.
