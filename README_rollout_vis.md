# SMART rollout visualization patch

## What changed

- `smart/modules/agent_flow_decoder.py`
  - returns `pred_valid_mask` from closed-loop inference.
- `smart/model/smart.py`
  - during validation/eval, saves rollout visualization for a selected scenario.
- `smart/utils/rollout_visualizer.py`
  - renders map + agents + trajectories to PNG frames and MP4/GIF.
- `eval.py`
  - adds CLI overrides for rollout visualization.
- `configs/validation/validation_flow.yaml`
  - adds a `Model.visualization` section.

## Supported options

- `trajectory_mode`: `point`, `line`, `box` (`rectangle` also accepted from CLI)
- `save_gap_sec`: frame save interval in seconds
- `scenario_index_in_batch`: which scenario inside the batch to visualize
- `max_scenarios`: how many scenarios to save during one evaluation run

## Example

```bash
python eval.py \
  --config configs/validation/validation_flow.yaml \
  --pretrain_ckpt /path/to/checkpoint.ckpt \
  --rollout_vis \
  --rollout_vis_dir debug_rollout_vis \
  --rollout_vis_mode box \
  --rollout_vis_save_gap 0.1 \
  --rollout_vis_max_scenarios 1
```

## Output layout

```text
rollout_vis/
  <scenario_id>/
    frames/
      00000.png
      00001.png
      ...
    <scenario_id>.mp4
    <scenario_id>.gif
```

## Notes

- Map rendering distinguishes `lane`, `road_line`, `road_edge`, and `crosswalk`.
- Agent rendering distinguishes `vehicle`, `pedestrian`, and `cyclist`.
- Past / generated / GT trajectories are drawn together in each saved frame.
- MP4/GIF writing uses `imageio`. If your environment does not have it, install `imageio imageio-ffmpeg`.
