from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from smart.datamodules import MultiDataModule
from smart.model import SMART
from smart.utils.config import load_config_act
from smart.utils.log import Logging


if __name__ == '__main__':
    parser = ArgumentParser()
    predictor_hash = {"smart": SMART}
    parser.add_argument('--config', type=str, default='configs/train/train_flow.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--save_ckpt_path', type=str, default='')
    args = parser.parse_args()

    config = load_config_act(args.config)
    Predictor = predictor_hash[config.Model.predictor]
    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    datamodule = MultiDataModule(**vars(config.Dataset))
    model = Predictor(config.Model)
    if args.pretrain_ckpt != '':
        logger = Logging().log(level='DEBUG')
        model.load_params_from_file(filename=args.pretrain_ckpt, logger=logger)

    trainer_config = config.Trainer
    if getattr(config.Model, 'train_stage', 'open_loop') == 'open_loop':
        monitor_name = 'val_flow_loss'
        monitor_mode = 'min'
    else:
        monitor_name = 'val_minADE'
        monitor_mode = 'min'

    model_checkpoint = ModelCheckpoint(
        dirpath=args.save_ckpt_path,
        filename='{epoch:02d}',
        monitor=monitor_name,
        every_n_epochs=1,
        save_top_k=5,
        mode=monitor_mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy=strategy,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        num_nodes=trainer_config.num_nodes,
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=trainer_config.max_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
    )

    if args.ckpt_path == '':
        trainer.fit(model, datamodule)
    else:
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)