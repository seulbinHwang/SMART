from __future__ import annotations

import contextlib
import math
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData

from smart.metrics import minADE, minFDE
from smart.modules import SMARTDecoder
from smart.utils.flow_traj import boundary_consistency_loss, build_linear_flow_path, chunk_future_21_to_4x6
from torch.optim.lr_scheduler import LambdaLR


class SMART(pl.LightningModule):
    """SMART flow-matching 모델.

    기존 SMART의 map encoder, token library, 전처리 루틴은 그대로 재사용한다.
    바뀌는 것은 agent 미래 생성 loss와 rollout 방식뿐이다.

    Notes:
        - 공식 공개 설정과 맞추기 위해 train/val batch_size=1을 전제로 한다.
        - open-loop는 scene당 random anchor 1개를 기본으로 사용한다.
        - closed-loop는 4번 0.5초 self-feeding unroll을 수행한다.
    """

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
        self.train_stage = getattr(model_config, "train_stage", "open_loop")
        self.flow_ode_steps = int(getattr(model_config, "flow_ode_steps", 4))
        self.closed_loop_unroll = int(getattr(model_config, "closed_loop_unroll", 4))
        self.inference_flow = True
        self.noise = True

        module_dir = os.path.dirname(os.path.dirname(__file__))
        self.map_token_traj_path = os.path.join(module_dir, "tokens/map_traj_token5.pkl")
        self.token_path = os.path.join(module_dir, "tokens/cluster_frame_5_2048.pkl")
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
            map_token={"traj_src": self.map_token["traj_src"]},
            token_data=token_data,
            token_size=model_config.decoder.token_size,
        )
        # agent decoder 내부 ODE step을 config와 맞춘다.
        self.encoder.agent_encoder.ode_steps = self.flow_ode_steps

        self.minADE = minADE(max_guesses=1)
        self.minFDE = minFDE(max_guesses=1)
        self.flow_loss_fn = nn.MSELoss(reduction="none")

    # ------------------------------------------------------------------
    # token file helpers
    # ------------------------------------------------------------------
    def get_trajectory_token(self) -> Dict:
        """agent motion token library를 읽는다."""
        token_data = pickle.load(open(self.token_path, "rb"))
        self.trajectory_token = token_data["token"]
        self.trajectory_token_traj = token_data["traj"]
        self.trajectory_token_all = token_data["token_all"]
        return token_data

    def init_map_token(self) -> None:
        """map token library를 읽고 샘플 포인트를 만든다."""
        self.argmin_sample_len = 3
        map_token_traj = pickle.load(open(self.map_token_traj_path, "rb"))
        self.map_token = {"traj_src": map_token_traj["traj_src"]}
        traj_end_theta = np.arctan2(
            self.map_token["traj_src"][:, -1, 1] - self.map_token["traj_src"][:, -2, 1],
            self.map_token["traj_src"][:, -1, 0] - self.map_token["traj_src"][:, -2, 0],
        )
        indices = torch.linspace(0, self.map_token["traj_src"].shape[1] - 1, steps=self.argmin_sample_len).long()
        self.map_token["sample_pt"] = torch.from_numpy(self.map_token["traj_src"][:, indices]).to(torch.float)
        self.map_token["traj_end_theta"] = torch.from_numpy(traj_end_theta).to(torch.float)
        self.map_token["traj_src"] = torch.from_numpy(self.map_token["traj_src"]).to(torch.float)

    # ------------------------------------------------------------------
    # model forward
    # ------------------------------------------------------------------
    def forward(self, data: HeteroData):
        return self.encoder(data)

    def inference(self, data: HeteroData):
        return self.encoder.inference(data)

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # original SMART map token preprocessing (unchanged)
    # ------------------------------------------------------------------
    def match_token_map(self, data: HeteroData) -> HeteroData:
        traj_pos = data["map_save"]["traj_pos"].to(torch.float)
        traj_theta = data["map_save"]["traj_theta"].to(torch.float)
        pl_idx_list = data["map_save"]["pl_idx_list"]
        token_sample_pt = self.map_token["sample_pt"].to(traj_pos.device)
        token_src = self.map_token["traj_src"].to(traj_pos.device)
        max_traj_len = self.map_token["traj_src"].shape[1]
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
            left_side = (data["pt_token"]["side"][pt] == 0).sum()
            right_side = (data["pt_token"]["side"][pt] == 1).sum()
            center_side = (data["pt_token"]["side"][pt] == 2).sum()
            count_nums.append(torch.tensor([left_side, right_side, center_side], device=traj_pos.device))
        count_nums = torch.stack(count_nums, dim=0)
        num_polyline = int(count_nums.max().item())
        traj_mask = torch.zeros((int(len(pl_idx_full.unique())), 3, num_polyline), dtype=torch.bool, device=traj_pos.device)
        idx_matrix = torch.arange(traj_mask.size(2), device=traj_pos.device).unsqueeze(0).unsqueeze(0)
        idx_matrix = idx_matrix.expand(traj_mask.size(0), traj_mask.size(1), -1)
        counts_num_expanded = count_nums.unsqueeze(-1)
        mask_update = idx_matrix < counts_num_expanded
        traj_mask[mask_update] = True

        data["pt_token"]["traj_mask"] = traj_mask
        data["pt_token"]["position"] = torch.cat(
            [pt_token_pos, torch.zeros((data["pt_token"]["num_nodes"], 1), device=traj_pos.device, dtype=torch.float)],
            dim=-1,
        )
        data["pt_token"]["orientation"] = pt_token_orientation
        data["pt_token"]["height"] = data["pt_token"]["position"][:, -1]
        data[("pt_token", "to", "map_polygon")] = {}
        data[("pt_token", "to", "map_polygon")]["edge_index"] = token2pl
        data["pt_token"]["token_idx"] = pt_token_id
        return data

    def sample_pt_pred(self, data: HeteroData) -> HeteroData:
        """원래 SMART의 map-token prediction mask 생성 루틴을 그대로 쓴다."""
        traj_mask = data["pt_token"]["traj_mask"]
        raw_pt_index = torch.arange(1, traj_mask.shape[2], device=traj_mask.device).repeat(traj_mask.shape[0], traj_mask.shape[1], 1)
        masked_pt_index = raw_pt_index.view(-1)[
            torch.randperm(raw_pt_index.numel(), device=traj_mask.device)[
                : traj_mask.shape[0] * traj_mask.shape[1] * ((traj_mask.shape[2] - 1) // 3)
            ].reshape(traj_mask.shape[0], traj_mask.shape[1], (traj_mask.shape[2] - 1) // 3)
        ]
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

        data["pt_token"]["pt_valid_mask"] = pt_valid_mask[traj_mask]
        data["pt_token"]["pt_pred_mask"] = pt_pred_mask[traj_mask]
        data["pt_token"]["pt_target_mask"] = pt_target_mask[traj_mask]
        return data

    # ------------------------------------------------------------------
    # flow target helpers
    # ------------------------------------------------------------------
    def _sample_anchor_raw(self, closed_loop: bool = False) -> int:
        """scene당 하나의 raw anchor를 뽑는다.

        공식 공개 설정과 같은 batch_size=1 전제이므로 scene 내부에서 하나만 뽑는다.

        Returns:
            raw step index. history 끝 시점 10을 기준으로 0.5초 간격으로 이동한다.
        """
        if closed_loop:
            valid = list(range(0, 46, 5))  # 0.0 .. 4.5s start for 4-step unroll
        else:
            valid = list(range(0, 61, 5))  # 0.0 .. 6.0s for full 2s target
        sampled = int(valid[torch.randint(0, len(valid), (1,)).item()])
        return (self.num_historical_steps - 1) + sampled

    def _local_future_from_anchor_pose(
        self,
        pos_world: torch.Tensor,
        heading_world: torch.Tensor,
        anchor_pos_world: torch.Tensor,
        anchor_heading_world: torch.Tensor,
    ) -> torch.Tensor:
        """GT world trajectory를 anchor local frame 상태열로 바꾼다.

        Args:
            pos_world: [A, 21, 2]
            heading_world: [A, 21]
            anchor_pos_world: [A, 2]
            anchor_heading_world: [A]

        Returns:
            [A, 21, 4]
        """
        rel = pos_world - anchor_pos_world[:, None]
        cos_h = torch.cos(anchor_heading_world)[:, None]
        sin_h = torch.sin(anchor_heading_world)[:, None]
        xl = rel[..., 0] * cos_h + rel[..., 1] * sin_h
        yl = -rel[..., 0] * sin_h + rel[..., 1] * cos_h
        dhead = torch.atan2(torch.sin(heading_world - anchor_heading_world[:, None]), torch.cos(heading_world - anchor_heading_world[:, None]))
        return torch.stack([xl, yl, torch.sin(dhead), torch.cos(dhead)], dim=-1)

    def prepare_open_loop_flow(self, data: HeteroData) -> HeteroData:
        """scene당 random anchor 1개를 뽑아 flow target을 만든다.

        Returns:
            data 내부에 아래 field를 추가한다.
            - flow_anchor_raw: scalar tensor
            - flow_anchor_slot: scalar tensor
            - flow_gt_future: [A, 21, 4]
            - flow_gt_segments: [A, 4, 6, 4]
            - flow_target_mask: [A]
            - flow_noisy_segments: [A, 4, 6, 4]
            - flow_t: [A]
        """
        if isinstance(data, Batch) and data.num_graphs > 1:
            raise NotImplementedError("현재 구현은 공개 설정과 같은 batch_size=1을 전제로 한다.")
        anchor_raw = self._sample_anchor_raw(closed_loop=False)
        anchor_slot = anchor_raw // self.encoder.agent_encoder.shift if hasattr(self.encoder, 'agent_encoder') else anchor_raw // 5
        # fallback for direct model-level use
        if not hasattr(self, "encoder"):
            anchor_slot = anchor_raw // 5

        pos_world = data["agent"]["position"][:, anchor_raw : anchor_raw + 21, : self.input_dim].contiguous()
        heading_world = data["agent"]["heading"][:, anchor_raw : anchor_raw + 21].contiguous()
        anchor_pos_world = data["agent"]["position"][:, anchor_raw, : self.input_dim].contiguous()
        anchor_heading_world = data["agent"]["heading"][:, anchor_raw].contiguous()
        flow_gt = self._local_future_from_anchor_pose(pos_world, heading_world, anchor_pos_world, anchor_heading_world)
        flow_gt_segments = chunk_future_21_to_4x6(flow_gt)
        valid = data["agent"]["valid_mask"][:, anchor_raw : anchor_raw + 21].all(dim=1)
        target_mask = valid & (data["agent"]["category"] == 3)
        noise = torch.randn_like(flow_gt_segments)
        t = torch.rand(flow_gt_segments.size(0), device=flow_gt_segments.device)
        flow_noisy = build_linear_flow_path(flow_gt_segments, noise, t)
        data["agent"]["flow_anchor_raw"] = torch.tensor(anchor_raw, device=flow_gt.device)
        data["agent"]["flow_anchor_slot"] = torch.tensor(anchor_slot, device=flow_gt.device)
        data["agent"]["flow_gt_future"] = flow_gt
        data["agent"]["flow_gt_segments"] = flow_gt_segments
        data["agent"]["flow_target_mask"] = target_mask
        data["agent"]["flow_noisy_segments"] = flow_noisy
        data["agent"]["flow_t"] = t
        return data

    def _flow_losses(self, pred: Dict[str, torch.Tensor], data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """flow / boundary loss를 계산한다."""
        gt = data["agent"]["flow_gt_segments"]
        mask = pred["flow_target_mask"]
        if not torch.any(mask):
            zero = gt.new_zeros(())
            return zero, zero, zero
        flow_res = self.flow_loss_fn(pred["flow_pred_segments"][mask], gt[mask]).mean()
        boundary = boundary_consistency_loss(pred["flow_pred_segments"][mask])
        total = flow_res + boundary
        return total, flow_res, boundary

    # ------------------------------------------------------------------
    # train / val
    # ------------------------------------------------------------------
    def training_step(self, data: HeteroData, batch_idx: int):
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

        if self.train_stage == "open_loop":
            data = self.prepare_open_loop_flow(data)
            pred = self(data)
            loss, flow_loss, boundary_loss = self._flow_losses(pred, data)
        else:
            loss, flow_loss, boundary_loss = self._closed_loop_training_loss(data)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log("train_flow_loss", flow_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log("train_boundary_loss", boundary_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        return loss

    @torch.no_grad()
    def _build_closed_loop_targets(
        self,
        data: HeteroData,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """현재 rollout state 기준 local GT 2.0초를 만든다.

        Args:
            state: agent decoder rollout state.

        Returns:
            gt_segments: [A, 4, 6, 4]
            mask: [A]
        """
        anchor_raw = int(state["current_raw"].item())
        pos_world = data["agent"]["position"][:, anchor_raw : anchor_raw + 21, : self.input_dim].contiguous()
        heading_world = data["agent"]["heading"][:, anchor_raw : anchor_raw + 21].contiguous()
        gt_local = self._local_future_from_anchor_pose(pos_world, heading_world, state["current_pos_world"], state["current_heading_world"])
        gt_segments = chunk_future_21_to_4x6(gt_local)
        mask = data["agent"]["valid_mask"][:, anchor_raw : anchor_raw + 21].all(dim=1) & (data["agent"]["category"] == 3)
        return gt_segments, mask

    def _closed_loop_training_loss(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """짧은 self-feeding closed-loop loss를 계산한다.

        구현을 과도하게 복잡하게 만들지 않기 위해, state update는 no_grad로 수행한다.
        gradient는 각 현재 step의 clean prediction loss에만 건다.
        """
        map_enc = self.encoder.map_encoder(data)
        start_anchor_raw = self._sample_anchor_raw(closed_loop=True)
        state = self.encoder.agent_encoder.build_initial_rollout_state(data, anchor_raw=start_anchor_raw)
        total = data["agent"]["position"].new_zeros(())
        total_flow = data["agent"]["position"].new_zeros(())
        total_boundary = data["agent"]["position"].new_zeros(())
        for _ in range(self.closed_loop_unroll):
            gt_segments, mask = self._build_closed_loop_targets(data, state)
            noise = torch.randn_like(gt_segments)
            t = torch.rand(gt_segments.size(0), device=gt_segments.device)
            z_u = build_linear_flow_path(gt_segments, noise, t)
            pred_clean = self.encoder.agent_encoder._predict_clean_segments_from_state(
                data=data,
                map_enc=map_enc,
                token_pos=state["token_pos"],
                token_heading=state["token_heading"],
                token_idx=state["token_idx"],
                current_slot=int(state["current_slot"].item()),
                cur_pos_world=state["current_pos_world"],
                cur_heading_world=state["current_heading_world"],
                cur_pos_prev_world=state["prev_pos_world"],
                cur_heading_prev_world=state["prev_heading_world"],
                noisy_segments=z_u,
                flow_t=t,
                valid_agent_mask=state["valid_now_mask"],
            )
            if torch.any(mask):
                flow_loss = self.flow_loss_fn(pred_clean[mask], gt_segments[mask]).mean()
                boundary = boundary_consistency_loss(pred_clean[mask])
            else:
                flow_loss = pred_clean.new_zeros(())
                boundary = pred_clean.new_zeros(())
            total = total + flow_loss + boundary
            total_flow = total_flow + flow_loss
            total_boundary = total_boundary + boundary
            with torch.no_grad():
                # rollout state update는 detach한다.
                self.encoder.agent_encoder.rollout_step(data, map_enc, state)
        scale = 1.0 / float(self.closed_loop_unroll)
        return total * scale, total_flow * scale, total_boundary * scale

    def validation_step(self, data: HeteroData, batch_idx: int):
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

        data = self.prepare_open_loop_flow(data)
        pred = self(data)
        loss, flow_loss, boundary_loss = self._flow_losses(pred, data)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val_flow_loss", flow_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val_boundary_loss", boundary_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        if self.inference_flow:
            pred_inf = self.inference(data)
            eval_mask = data["agent"]["valid_mask"][:, self.num_historical_steps - 1]
            valid_mask = data["agent"]["valid_mask"][:, self.num_historical_steps :]
            self.minADE.update(pred=pred_inf["pred_traj"][eval_mask], target=pred_inf["gt"][eval_mask], valid_mask=valid_mask[eval_mask])
            self.minFDE.update(pred=pred_inf["pred_traj"][eval_mask], target=pred_inf["gt"][eval_mask], valid_mask=valid_mask[eval_mask])
            self.log("val_minADE", self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log("val_minFDE", self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

    # ------------------------------------------------------------------
    # optimizer / checkpoint
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step: int):
            if current_step + 1 < self.warmup_steps:
                return float(current_step + 1) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                0.5
                * (
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
        """기존 SMART checkpoint 로더를 그대로 쓴다."""
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info("==> Loading parameters from checkpoint %s to %s" % (filename, "CPU" if to_cpu else "GPU"))
        loc_type = torch.device("cpu") if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint["state_dict"]
        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk_filter, strict=False)
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")
        epoch = checkpoint.get("epoch", -1)
        it = checkpoint.get("it", 0.0)
        return it, epoch
