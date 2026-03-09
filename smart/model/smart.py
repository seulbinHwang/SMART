from __future__ import annotations

import contextlib
import math
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData
from torch.optim.lr_scheduler import LambdaLR

from smart.metrics import AverageMeter, minADE, minFDE
from smart.modules import SMARTDecoder
from smart.utils.rollout_visualizer import build_rollout_visualization_config, render_rollout_visualization


class SMART(pl.LightningModule):
    """SMART backbone 위에 flow-based agent head를 올린 Lightning 모델."""

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.warmup_steps = model_config.warmup_steps
        self.lr = model_config.lr
        self.total_steps = model_config.total_steps
        self.dataset = model_config.dataset
        self.input_dim = model_config.input_dim
        self.hidden_dim = model_config.hidden_dim
        self.output_dim = model_config.output_dim
        self.output_head = model_config.output_head
        self.num_historical_steps = model_config.num_historical_steps
        self.num_future_steps = model_config.decoder.num_future_steps
        self.num_freq_bands = model_config.num_freq_bands
        self.future_window_steps = getattr(model_config.decoder, 'future_window_steps', 20)
        self.anchor_chunk_k = getattr(model_config.decoder, 'anchor_chunk_k', 4)
        self.ode_steps = getattr(model_config.decoder, 'ode_steps', 4)
        self.overlap_loss_weight = getattr(model_config.decoder, 'overlap_loss_weight', 0.1)
        self.closed_loop_steps = int(getattr(model_config.decoder, 'closed_loop_steps', 0))
        self.closed_loop_eval = getattr(model_config.decoder, 'closed_loop_eval', True)
        self.vis_map = False
        self.noise = True

        module_dir = os.path.dirname(os.path.dirname(__file__))
        self.map_token_traj_path = os.path.join(module_dir, 'tokens/map_traj_token5.pkl')
        self.token_path = os.path.join(module_dir, 'tokens/cluster_frame_5_2048.pkl')
        self.init_map_token()
        token_data = self.get_trajectory_token()

        self.encoder = SMARTDecoder(
            dataset=model_config.dataset,
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            num_historical_steps=model_config.num_historical_steps,
            num_freq_bands=model_config.num_freq_bands,
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            dropout=model_config.dropout,
            num_map_layers=model_config.decoder.num_map_layers,
            num_agent_layers=model_config.decoder.num_agent_layers,
            pl2pl_radius=model_config.decoder.pl2pl_radius,
            pl2a_radius=model_config.decoder.pl2a_radius,
            a2a_radius=model_config.decoder.a2a_radius,
            time_span=model_config.decoder.time_span,
            map_token={'traj_src': self.map_token['traj_src']},
            token_data=token_data,
            token_size=model_config.decoder.token_size,
            future_window_steps=self.future_window_steps,
            anchor_chunk_k=self.anchor_chunk_k,
            ode_steps=self.ode_steps,
        )
        self.minADE = minADE(max_guesses=1)
        self.minFDE = minFDE(max_guesses=1)
        self.val_open_loop_ade = AverageMeter()
        self.val_overlap = AverageMeter()
        self.val_flow = AverageMeter()
        self.rollout_vis_config = build_rollout_visualization_config(model_config)
        self._rollout_vis_saved = 0

    def get_trajectory_token(self) -> Dict:
        """SMART agent token 사전을 읽는다.

        Returns:
            token 파일 전체 dict.
        """
        token_data = pickle.load(open(self.token_path, 'rb'))
        self.trajectory_token = token_data['token']
        self.trajectory_token_traj = token_data['traj']
        self.trajectory_token_all = token_data['token_all']
        return token_data

    def init_map_token(self) -> None:
        """SMART map token 사전의 샘플 점을 준비한다."""
        self.argmin_sample_len = 3
        map_token_traj = pickle.load(open(self.map_token_traj_path, 'rb'))
        self.map_token = {'traj_src': map_token_traj['traj_src']}
        traj_end_theta = np.arctan2(
            self.map_token['traj_src'][:, -1, 1] - self.map_token['traj_src'][:, -2, 1],
            self.map_token['traj_src'][:, -1, 0] - self.map_token['traj_src'][:, -2, 0],
        )
        indices = torch.linspace(0, self.map_token['traj_src'].shape[1] - 1, steps=self.argmin_sample_len).long()
        self.map_token['sample_pt'] = torch.from_numpy(self.map_token['traj_src'][:, indices]).to(torch.float)
        self.map_token['traj_end_theta'] = torch.from_numpy(traj_end_theta).to(torch.float)
        self.map_token['traj_src'] = torch.from_numpy(self.map_token['traj_src']).to(torch.float)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """map encoder와 flow agent decoder를 함께 실행한다.

        Args:
            data: SMART 입력 HeteroData.

        Returns:
            flow 학습과 검증에 필요한 tensor 묶음.
        """
        return self.encoder(data)

    def inference(
        self,
        data: HeteroData,
        rollout_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """closed-loop rollout 추론을 수행한다.

        Args:
            data: SMART 입력 HeteroData.
            rollout_steps: 실제로 굴릴 raw step 수.
                None이면 전체 8초를 끝까지 굴린다.

        Returns:
            rollout 예측 결과 dict.
        """
        return self.encoder.inference(data, rollout_steps=rollout_steps)

    def maybe_autocast(self, dtype: torch.dtype = torch.float16):
        """GPU일 때만 autocast를 켠다.

        Args:
            dtype: autocast 계산 dtype.

        Returns:
            context manager.
        """
        if self.device != torch.device('cpu'):
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def _empty_loss(self) -> torch.Tensor:
        """유효 target이 하나도 없을 때 사용할 0 loss를 만든다.

        Returns:
            gradient가 끊기지 않는 scalar tensor.
        """
        return self.encoder.agent_encoder.segment_out_head.mlp[0].weight.sum() * 0.0

    def _compute_flow_losses(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """flow matching 학습 손실을 계산한다.

        Args:
            pred: decoder 출력 dict.

        Returns:
            loss dict.
        """
        if pred['flow_pred'].numel() == 0:
            zero = self._empty_loss()
            return {
                'loss': zero,
                'flow_loss': zero,
                'overlap_loss': zero,
                'open_loop_ade': zero.detach(),
            }
        flow_loss = ((pred['flow_pred'] - pred['flow_gt']) ** 2).mean()
        overlap_loss = pred['overlap_error'].mean()
        loss = flow_loss + self.overlap_loss_weight * overlap_loss
        open_loop_ade = pred['open_loop_ade'].mean()
        return {
            'loss': loss,
            'flow_loss': flow_loss,
            'overlap_loss': overlap_loss,
            'open_loop_ade': open_loop_ade,
        }


    def _compute_rollout_loss(
        self,
        data: HeteroData,
        rollout: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """짧은 closed-loop fine-tuning용 rollout loss를 계산한다.

        Args:
            data: 현재 batch HeteroData.
            rollout: `inference()` 결과 dict.

        Returns:
            torch.Tensor: scalar rollout loss.
        """
        eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1] & (data['agent']['type'] != 3)
        if int(eval_mask.sum().item()) == 0:
            return self._empty_loss()
        pred = rollout['pred_traj'][eval_mask]
        gt = rollout['gt'][eval_mask]
        valid = rollout['valid_mask'][eval_mask]
        weight = valid.unsqueeze(-1).to(pred.dtype)
        denom = (weight.sum() * pred.shape[-1]).clamp_min(1.0)
        return (((pred - gt) ** 2) * weight).sum() / denom


    def _batch_size(self, data) -> int:
        """현재 batch 안의 scene 수를 구한다.

        Args:
            data: `HeteroData` 또는 `Batch`.

        Returns:
            int: batch 안의 scene 수.
        """
        if isinstance(data, Batch):
            return int(data.num_graphs)
        return 1

    def _should_save_rollout_visualization(self) -> bool:
        return (
            bool(self.rollout_vis_config.enabled)
            and int(getattr(self, 'global_rank', 0)) == 0
            and self._rollout_vis_saved < int(self.rollout_vis_config.max_scenarios)
        )

    def _prepare_rollout_visualization_inputs(self, data):
        if not self._should_save_rollout_visualization():
            return None, None

        if isinstance(data, Batch):
            data_list = data.to_data_list()
            if len(data_list) == 0:
                return None, None
            scenario_index = min(
                max(int(self.rollout_vis_config.scenario_index_in_batch), 0),
                len(data_list) - 1,
            )
            agent_ptr = data['agent']['ptr']
            agent_range = (int(agent_ptr[scenario_index].item()), int(agent_ptr[scenario_index + 1].item()))
            return data_list[scenario_index], agent_range

        num_nodes = data['agent']['num_nodes']
        if isinstance(num_nodes, torch.Tensor):
            num_nodes = int(num_nodes.item())
        else:
            num_nodes = int(num_nodes)
        return data.clone(), (0, num_nodes)

    def _slice_rollout_for_visualization(self, rollout, agent_range):
        if rollout is None or agent_range is None:
            return None
        start, end = agent_range
        return {
            'pred_traj': rollout['pred_traj'][start:end],
            'pred_head': rollout['pred_head'][start:end],
            'pred_valid_mask': rollout.get('pred_valid_mask', None)[start:end]
            if rollout.get('pred_valid_mask', None) is not None
            else None,
            'gt': rollout['gt'][start:end],
            'valid_mask': rollout['valid_mask'][start:end],
        }

    def _maybe_save_rollout_visualization(self, scenario_data, rollout, batch_idx: int) -> None:
        if scenario_data is None or rollout is None or not self._should_save_rollout_visualization():
            return
        render_rollout_visualization(
            scenario_data=scenario_data,
            rollout=rollout,
            config=self.rollout_vis_config,
            batch_idx=int(batch_idx),
        )
        self._rollout_vis_saved += 1

    def training_step(self, data, batch_idx):
        """한 step의 open-loop flow 학습을 수행한다.

        Args:
            data: batch HeteroData.
            batch_idx: Lightning batch index.

        Returns:
            scalar loss tensor.
        """
        del batch_idx
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        losses = self._compute_flow_losses(pred)
        batch_size = self._batch_size(data)
        if self.closed_loop_steps > 0:
            rollout = self.inference(data, rollout_steps=self.closed_loop_steps * self.encoder.agent_encoder.shift)
            rollout_loss = self._compute_rollout_loss(data, rollout)
            losses['loss'] = losses['loss'] + rollout_loss
            self.log('train_rollout_loss', rollout_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_loss', losses['loss'], prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_flow_loss', losses['flow_loss'], prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_overlap_loss', losses['overlap_loss'], prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_open_loop_ade', losses['open_loop_ade'], prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return losses['loss']

    def validation_step(self, data, batch_idx):
        """open-loop 검증과 선택적 closed-loop rollout 검증을 수행한다.

        Args:
            data: batch HeteroData.
            batch_idx: Lightning batch index.
        """
        vis_scenario_data, vis_agent_range = self._prepare_rollout_visualization_inputs(data)

        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        losses = self._compute_flow_losses(pred)
        batch_size = self._batch_size(data)

        self.val_flow.update(losses['flow_loss'].detach().view(1))
        self.val_overlap.update(losses['overlap_loss'].detach().view(1))
        self.val_open_loop_ade.update(losses['open_loop_ade'].detach().view(1))
        self.log('val_loss', losses['loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_flow_loss', self.val_flow, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_overlap_loss', self.val_overlap, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_open_loop_ade', self.val_open_loop_ade, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        rollout = None
        if self.closed_loop_eval or self._should_save_rollout_visualization():
            rollout = self.inference(data)

        if self.closed_loop_eval and rollout is not None:
            eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1] & (data['agent']['type'] != 3)
            self.minADE.update(
                pred=rollout['pred_traj'][eval_mask],
                target=rollout['gt'][eval_mask],
                valid_mask=rollout['valid_mask'][eval_mask],
            )
            self.minFDE.update(
                pred=rollout['pred_traj'][eval_mask],
                target=rollout['gt'][eval_mask],
                valid_mask=rollout['valid_mask'][eval_mask],
            )
            self.log('val_rollout_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val_rollout_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if rollout is not None:
            vis_rollout = self._slice_rollout_for_visualization(rollout, vis_agent_range)
            self._maybe_save_rollout_visualization(vis_scenario_data, vis_rollout, batch_idx=batch_idx)

    def on_validation_start(self) -> None:
        """검증 누적값을 초기화한다."""
        self.minADE.reset()
        self.minFDE.reset()
        self.val_flow.reset()
        self.val_overlap.reset()
        self.val_open_loop_ade.reset()
        self._rollout_vis_saved = 0

    def configure_optimizers(self):
        """optimizer와 cosine scheduler를 만든다.

        Returns:
            Lightning optimizer / scheduler tuple.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step: int) -> float:
            if current_step + 1 < self.warmup_steps:
                return float(current_step + 1) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * (current_step - self.warmup_steps)
                        / float(max(1, self.total_steps - self.warmup_steps))
                    )
                ),
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def load_params_from_file(self, filename, logger, to_cpu: bool = False):
        """기존 SMART 체크포인트를 가능한 범위까지만 불러온다.

        이름과 tensor shape가 둘 다 맞는 가중치만 로드한다. 그래서 이번처럼
        agent head를 바꾼 경우에도 map encoder와 공통 backbone 부분은 최대한
        재사용할 수 있다.

        Args:
            filename: checkpoint 경로.
            logger: logger 객체.
            to_cpu: True면 CPU로 로드.

        Returns:
            (iteration, epoch)
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['state_dict']

        version = checkpoint.get('version', None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(
                        f'Ignore key in disk (shape does not match): {key}, '
                        f'load_shape={val.shape}, model_shape={model_state[key].shape}'
                    )

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk_filter, strict=False)
        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'The number of missing keys: {len(missing_keys)}')
        logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        logger.info('==> Done (total keys %d)' % (len(model_state)))
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)
        return it, epoch

    def match_token_map(self, data: HeteroData):
        """원본 SMART map token matching 전처리를 그대로 수행한다.

        Args:
            data: SMART 입력 HeteroData.

        Returns:
            map token 정보가 채워진 HeteroData.
        """
        traj_pos = data['map_save']['traj_pos'].to(torch.float)
        traj_theta = data['map_save']['traj_theta'].to(torch.float)
        pl_idx_list = data['map_save']['pl_idx_list']
        token_sample_pt = self.map_token['sample_pt'].to(traj_pos.device)
        token_src = self.map_token['traj_src'].to(traj_pos.device)
        max_traj_len = self.map_token['traj_src'].shape[1]
        pl_num = traj_pos.shape[0]

        pt_token_pos = traj_pos[:, 0, :].clone()
        pt_token_orientation = traj_theta.clone()
        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = -sin
        rot_mat[..., 1, 0] = sin
        rot_mat[..., 1, 1] = cos
        traj_pos_local = torch.bmm((traj_pos - traj_pos[:, 0:1]), rot_mat.view(-1, 2, 2))
        distance = torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1)) ** 2, dim=(-2, -1))
        pt_token_id = torch.argmin(distance, dim=1)

        if self.noise:
            topk_indices = torch.argsort(
                torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1)) ** 2, dim=(-2, -1)),
                dim=1,
            )[:, :8]
            sample_topk = torch.randint(0, topk_indices.shape[-1], size=(topk_indices.shape[0], 1), device=topk_indices.device)
            pt_token_id = torch.gather(topk_indices, 1, sample_topk).squeeze(-1)

        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = sin
        rot_mat[..., 1, 0] = -sin
        rot_mat[..., 1, 1] = cos
        token_src_world = torch.bmm(
            token_src[None, ...].repeat(pl_num, 1, 1, 1).reshape(pl_num, -1, 2),
            rot_mat.view(-1, 2, 2),
        ).reshape(pl_num, token_src.shape[0], max_traj_len, 2) + traj_pos[:, None, [0], :]
        _ = token_src_world.view(-1, 1024, 11, 2)[torch.arange(pt_token_id.view(-1).shape[0]), pt_token_id.view(-1)]

        pl_idx_full = pl_idx_list.clone()
        token2pl = torch.stack([torch.arange(len(pl_idx_list), device=traj_pos.device), pl_idx_full.long()])
        count_nums = []
        for pl in pl_idx_full.unique():
            pt = token2pl[0, token2pl[1, :] == pl]
            left_side = (data['pt_token']['side'][pt] == 0).sum()
            right_side = (data['pt_token']['side'][pt] == 1).sum()
            center_side = (data['pt_token']['side'][pt] == 2).sum()
            count_nums.append(torch.tensor([left_side, right_side, center_side], device=traj_pos.device))
        count_nums = torch.stack(count_nums, dim=0)
        num_polyline = int(count_nums.max().item())
        traj_mask = torch.zeros((int(len(pl_idx_full.unique())), 3, num_polyline), dtype=bool, device=traj_pos.device)
        idx_matrix = torch.arange(traj_mask.size(2), device=traj_pos.device).unsqueeze(0).unsqueeze(0)
        idx_matrix = idx_matrix.expand(traj_mask.size(0), traj_mask.size(1), -1)
        counts_num_expanded = count_nums.unsqueeze(-1)
        mask_update = idx_matrix < counts_num_expanded
        traj_mask[mask_update] = True

        data['pt_token']['traj_mask'] = traj_mask
        data['pt_token']['position'] = torch.cat(
            [
                pt_token_pos,
                torch.zeros((data['pt_token']['num_nodes'], 1), device=traj_pos.device, dtype=torch.float),
            ],
            dim=-1,
        )
        data['pt_token']['orientation'] = pt_token_orientation
        data['pt_token']['height'] = data['pt_token']['position'][:, -1]
        data[('pt_token', 'to', 'map_polygon')] = {}
        data[('pt_token', 'to', 'map_polygon')]['edge_index'] = token2pl
        data['pt_token']['token_idx'] = pt_token_id
        return data

    def sample_pt_pred(self, data: HeteroData):
        """원본 SMART map reconstruction용 masking을 그대로 만든다.

        Args:
            data: SMART 입력 HeteroData.

        Returns:
            map point prediction mask가 추가된 HeteroData.
        """
        traj_mask = data['pt_token']['traj_mask']
        raw_pt_index = torch.arange(1, traj_mask.shape[2], device=traj_mask.device).repeat(
            traj_mask.shape[0], traj_mask.shape[1], 1
        )
        masked_pt_index = raw_pt_index.view(-1)[
            torch.randperm(raw_pt_index.numel(), device=traj_mask.device)[
                : traj_mask.shape[0] * traj_mask.shape[1] * ((traj_mask.shape[2] - 1) // 3)
            ]
        ].reshape(traj_mask.shape[0], traj_mask.shape[1], (traj_mask.shape[2] - 1) // 3)
        masked_pt_index = torch.sort(masked_pt_index, -1)[0]
        pt_valid_mask = traj_mask.clone()
        pt_valid_mask.scatter_(2, masked_pt_index, False)
        pt_pred_mask = traj_mask.clone()
        pt_pred_mask.scatter_(2, masked_pt_index, False)
        tmp_mask = pt_pred_mask.clone()
        tmp_mask[:, :, :] = True
        tmp_mask.scatter_(2, masked_pt_index - 1, False)
        pt_pred_mask.masked_fill_(tmp_mask, False)
        pt_pred_mask = pt_pred_mask * torch.roll(traj_mask, shifts=-1, dims=2)
        pt_target_mask = torch.roll(pt_pred_mask, shifts=1, dims=2)

        data['pt_token']['pt_valid_mask'] = pt_valid_mask[traj_mask]
        data['pt_token']['pt_pred_mask'] = pt_pred_mask[traj_mask]
        data['pt_token']['pt_target_mask'] = pt_target_mask[traj_mask]
        return data