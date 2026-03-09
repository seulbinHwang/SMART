from argparse import ArgumentParser

import easydict
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging


def _ensure_visualization_config(config) -> None:
    if not hasattr(config.Model, "visualization") or config.Model.visualization is None:
        config.Model.visualization = easydict.EasyDict(
            {
                "enabled": False,
                "output_dir": "rollout_vis",
                "scenario_index_in_batch": 0,
                "max_scenarios": 1,
                "save_gap_sec": 0.1,
                "raw_step_sec": 0.1,
                "trajectory_mode": "line",
                "figsize": (12.0, 12.0),
                "dpi": 180,
                "margin_m": 20.0,
                "video_fps": None,
                "draw_agent_ids": False,
            }
        )


def _apply_rollout_visualization_overrides(config, args) -> None:
    _ensure_visualization_config(config)
    vis_cfg = config.Model.visualization

    if args.rollout_vis:
        vis_cfg.enabled = True
    if args.rollout_vis_dir:
        vis_cfg.output_dir = args.rollout_vis_dir
    if args.rollout_vis_mode:
        vis_cfg.trajectory_mode = args.rollout_vis_mode
    if args.rollout_vis_save_gap is not None:
        vis_cfg.save_gap_sec = float(args.rollout_vis_save_gap)
    if args.rollout_vis_scenario_index is not None:
        vis_cfg.scenario_index_in_batch = int(args.rollout_vis_scenario_index)
    if args.rollout_vis_max_scenarios is not None:
        vis_cfg.max_scenarios = int(args.rollout_vis_max_scenarios)
    if args.rollout_vis_raw_step_sec is not None:
        vis_cfg.raw_step_sec = float(args.rollout_vis_raw_step_sec)
    if args.rollout_vis_video_fps is not None:
        vis_cfg.video_fps = int(args.rollout_vis_video_fps)
    if args.rollout_vis_draw_agent_ids:
        vis_cfg.draw_agent_ids = True


if __name__ == '__main__':

    pl.seed_everything(2, workers=True)

    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/validation/validation_flow.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--save_ckpt_path', type=str, default='')

    parser.add_argument('--rollout_vis', action='store_true')
    parser.add_argument('--rollout_vis_dir', type=str, default='')
    parser.add_argument('--rollout_vis_mode', type=str, default='')
    parser.add_argument('--rollout_vis_save_gap', type=float, default=None)
    parser.add_argument('--rollout_vis_scenario_index', type=int, default=None)
    parser.add_argument('--rollout_vis_max_scenarios', type=int, default=None)
    parser.add_argument('--rollout_vis_raw_step_sec', type=float, default=None)
    parser.add_argument('--rollout_vis_video_fps', type=int, default=None)
    parser.add_argument('--rollout_vis_draw_agent_ids', action='store_true')

    args = parser.parse_args()

    config = load_config_act(args.config)
    _apply_rollout_visualization_overrides(config, args)

    data_config = config.Dataset

    val_dataset = {
        'scalable': MultiDataset,
    }[data_config.dataset](
        root=data_config.root,
        split='val',
        raw_dir=data_config.val_raw_dir,
        processed_dir=data_config.val_processed_dir,
        transform=WaymoTargetBuilder(config.Model.num_historical_steps, config.Model.decoder.num_future_steps),
    )

    dataloader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=True if data_config.num_workers > 0 else False,
    )

    if args.pretrain_ckpt == '':
        model = SMART(config.Model)
    else:
        logger = Logging().log(level='DEBUG')
        model = SMART(config.Model)
        model.load_params_from_file(filename=args.pretrain_ckpt, logger=logger)

    trainer_config = config.Trainer

    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy='ddp',
        num_sanity_val_steps=0,
        precision=trainer_config.precision,
    )

    trainer.validate(model, dataloader)