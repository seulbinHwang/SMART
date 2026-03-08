from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_geometric.data import Batch, HeteroData

from smart.layers import MLPLayer
from smart.layers.attention_layer import AttentionLayer
from smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from smart.modules.agent_decoder import SMARTAgentDecoder
from smart.utils import (
    angle_between_2d_vectors,
    assemble_4x6_to_21,
    build_ot_flow_path,
    chunk_future_21_to_4x6,
    get_valid_anchor_indices,
    global_last_pose_from_local_segment,
    local_future_from_global,
    overlap_consistency_error,
    wrap_angle,
)


class SMARTAgentFlowDecoder(SMARTAgentDecoder):
    """SMART의 agent head를 sparse flow matching head로 바꾼 구현.

    이 클래스는 기존 SMARTAgentDecoder가 이미 가지고 있는

    * type / shape 임베딩
    * relation 임베딩
    * sparse temporal / map / a2a attention 블록

    을 최대한 재사용한다. 바꾸는 부분은 agent의 미래를 token 분류로 맞히는
    마지막 head뿐이다.

    미래 2.0초를 4개의 0.5초 조각으로 나누고, 각 조각을 noisy segment에서
    clean segment로 복원하는 conditional flow matching 방식으로 학습한다.
    """

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        token_data: Dict,
        token_size: int = 512,
        future_window_steps: int = 20,
        anchor_chunk_k: int = 4,
        ode_steps: int = 4,
        target_category: int = 3,
        flow_eps: float = 1e-3,
    ) -> None:
        super().__init__(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            token_data=token_data,
            token_size=token_size,
        )
        self.future_window_steps = future_window_steps
        self.anchor_chunk_k = anchor_chunk_k
        self.ode_steps = ode_steps
        self.target_category = target_category
        self.flow_eps = flow_eps
        self.history_slots = 6
        self.current_slot = 1
        self.future_segments = 4
        self.segment_points = 6
        self.state_dim = 4
        self.current_state_dim = 8

        self.current_anchor_emb = MLPEmbedding(input_dim=self.current_state_dim, hidden_dim=hidden_dim)
        self.future_segment_emb = MLPEmbedding(
            input_dim=self.segment_points * self.state_dim,
            hidden_dim=hidden_dim,
        )
        self.flow_time_emb = FourierEmbedding(input_dim=1, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.segment_index_emb = nn.Embedding(self.future_segments, hidden_dim)
        self.segment_out_head = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=self.segment_points * self.state_dim,
        )
        self.hist_mask = False
        self.r_hist_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.context_t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.future_t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.hist2fut_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

    def _agent_batch(self, data: HeteroData, mask: torch.Tensor) -> torch.Tensor:
        """선택된 agent들의 scene batch index를 돌려준다.

        Args:
            data: SMART 입력 HeteroData.
            mask: shape (A,) bool.

        Returns:
            shape (A_sel,) long tensor.
        """
        if isinstance(data, Batch):
            return data['agent']['batch'][mask]
        return torch.zeros(int(mask.sum().item()), dtype=torch.long, device=mask.device)

    def _select_anchor_indices(self, data: HeteroData) -> torch.Tensor:
        """이번 forward에서 사용할 anchor raw 시각을 고른다.

        학습 중에는 scene 내부 병렬 감독은 유지하되, 메모리 폭증을 막기 위해
        anchor 전체를 쓰지 않고 `anchor_chunk_k`개만 뽑는다. 검증 때는 가능한
        anchor를 모두 쓴다.

        Args:
            data: SMART 입력 HeteroData.

        Returns:
            shape (K,) long tensor.
        """
        total_steps = data['agent']['position'].shape[1]
        anchors = get_valid_anchor_indices(
            total_steps=total_steps,
            num_historical_steps=self.num_historical_steps,
            future_window_steps=self.future_window_steps,
            shift=self.shift,
            device=data['agent']['position'].device,
        )
        if self.training and anchors.numel() > self.anchor_chunk_k:
            perm = torch.randperm(anchors.numel(), device=anchors.device)[: self.anchor_chunk_k]
            anchors = anchors[perm]
            anchors, _ = anchors.sort()
        return anchors

    def _slice_agent_meta(self, data: HeteroData, mask: torch.Tensor) -> Dict:
        """선택된 agent subset만 담은 작은 입력 묶음을 만든다.

        Args:
            data: 원본 SMART 입력.
            mask: shape (A,) bool.

        Returns:
            `agent_token_embedding()`가 바로 받을 수 있는 작은 dict.
        """
        return {
            'agent': {
                'num_nodes': int(mask.sum().item()),
                'type': data['agent']['type'][mask],
                'shape': data['agent']['shape'][mask],
                'token_velocity': data['agent']['token_velocity'][mask],
            }
        }

    def _build_history_window(
        self,
        data: HeteroData,
        anchor_index: int,
        context_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """anchor 기준 과거 6개 SMART token slot을 꺼낸다.

        Args:
            data: SMART 입력.
            anchor_index: 현재 raw frame index.
            context_mask: shape (A,) bool. 현재 시각에 context로 둘 agent.

        Returns:
            hist_pos: shape (A_ctx, 6, 2)
            hist_heading: shape (A_ctx, 6)
            hist_token_idx: shape (A_ctx, 6)
            hist_mask: shape (A_ctx, 6)
            hist_raw_time: shape (A_ctx, 6)
        """
        token_pos = data['agent']['token_pos'][context_mask]
        token_heading = data['agent']['token_heading'][context_mask]
        token_idx = data['agent']['token_idx'][context_mask]
        token_valid = data['agent']['agent_valid_mask'][context_mask]
        anchor_token_index = anchor_index // self.shift - 1
        hist_indices = torch.arange(
            anchor_token_index - (self.history_slots - 1),
            anchor_token_index + 1,
            device=token_pos.device,
            dtype=torch.long,
        )
        hist_in_range = (hist_indices >= 0) & (hist_indices < token_pos.shape[1])
        hist_indices_clamped = hist_indices.clamp(0, token_pos.shape[1] - 1)

        hist_pos = token_pos[:, hist_indices_clamped]
        hist_heading = token_heading[:, hist_indices_clamped]
        hist_token_idx = token_idx[:, hist_indices_clamped]
        hist_mask = token_valid[:, hist_indices_clamped] & hist_in_range.unsqueeze(0)
        hist_raw_time = ((hist_indices.float() + 1.0) * float(self.shift)).unsqueeze(0).repeat(hist_pos.shape[0], 1)
        return hist_pos, hist_heading, hist_token_idx, hist_mask, hist_raw_time

    def _build_current_anchor_feature(
        self,
        data: HeteroData,
        anchor_index: int,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """현재 정확한 상태를 continuous anchor token으로 바꾼다.

        Args:
            data: SMART 입력.
            anchor_index: 현재 raw frame index.
            mask: shape (A,) bool.

        Returns:
            current_feat: shape (A_sel, 128)
            current_pos: shape (A_sel, 2)
            current_heading: shape (A_sel,)
            current_state: shape (A_sel, 8)
        """
        pos = data['agent']['position'][mask, anchor_index, :2]
        heading = data['agent']['heading'][mask, anchor_index]

        if data['agent']['velocity'].shape[-1] >= 2:
            vel_global = data['agent']['velocity'][mask, anchor_index, :2]
        else:
            vel_global = torch.zeros(pos.shape[0], 2, device=pos.device, dtype=pos.dtype)
        if anchor_index > 0:
            prev_heading = data['agent']['heading'][mask, anchor_index - 1]
            yaw_rate = wrap_angle(heading - prev_heading) / 0.1
        else:
            yaw_rate = torch.zeros_like(heading)

        cos = heading.cos()
        sin = heading.sin()
        rot = torch.zeros(pos.shape[0], 2, 2, device=pos.device, dtype=pos.dtype)
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = -sin
        rot[:, 1, 0] = sin
        rot[:, 1, 1] = cos
        vel_local = torch.bmm(vel_global.unsqueeze(1), rot).squeeze(1)

        if data['agent']['shape'].dim() == 3:
            shape = data['agent']['shape'][mask, anchor_index, :]
        else:
            shape = data['agent']['shape'][mask]
        if shape.shape[-1] < 3:
            pad = torch.zeros(shape.shape[0], 3 - shape.shape[-1], device=shape.device, dtype=shape.dtype)
            shape = torch.cat([shape, pad], dim=-1)

        agent_type = data['agent']['type'][mask].float().unsqueeze(-1)
        current_state = torch.cat(
            [
                vel_local,
                heading.sin().unsqueeze(-1),
                heading.cos().unsqueeze(-1),
                yaw_rate.unsqueeze(-1),
                shape[:, 0:1],
                shape[:, 1:2],
                agent_type,
            ],
            dim=-1,
        )
        categorical = [
            self.type_a_emb(data['agent']['type'][mask].long()),
            self.shape_emb(shape),
        ]
        current_feat = self.current_anchor_emb(current_state) + torch.stack(categorical).sum(dim=0)
        return current_feat, pos, heading, current_state

    def _encode_context(
        self,
        data: HeteroData,
        anchor_index: int,
        context_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """scene-wide 과거 memory를 만든다.

        Args:
            data: SMART 입력.
            anchor_index: 현재 raw frame index.
            context_mask: shape (A,) bool.

        Returns:
            context에 필요한 tensor 묶음.
        """
        hist_pos, hist_heading, hist_token_idx, hist_mask, hist_raw_time = self._build_history_window(
            data=data,
            anchor_index=anchor_index,
            context_mask=context_mask,
        )
        current_feat, current_pos, current_heading, _ = self._build_current_anchor_feature(
            data=data,
            anchor_index=anchor_index,
            mask=context_mask,
        )
        context_meta = self._slice_agent_meta(data, context_mask)
        hist_head_vector = torch.stack([hist_heading.cos(), hist_heading.sin()], dim=-1)
        hist_feat, _ = self.agent_token_embedding(
            data=context_meta,
            agent_category=torch.zeros(hist_pos.shape[0], device=hist_pos.device),
            agent_token_index=hist_token_idx,
            pos_a=hist_pos,
            head_vector_a=hist_head_vector,
            inference=False,
        )
        hist_feat = hist_feat * hist_mask.unsqueeze(-1)
        context_feat = torch.cat([hist_feat, current_feat.unsqueeze(1)], dim=1)
        context_pos = torch.cat([hist_pos, current_pos.unsqueeze(1)], dim=1)
        context_heading = torch.cat([hist_heading, current_heading.unsqueeze(1)], dim=1)
        context_mask_full = torch.cat(
            [
                hist_mask,
                torch.ones(hist_mask.shape[0], 1, dtype=torch.bool, device=hist_mask.device),
            ],
            dim=1,
        )
        context_raw_time = torch.cat(
            [
                hist_raw_time,
                torch.full(
                    (hist_pos.shape[0], 1),
                    float(anchor_index),
                    device=hist_pos.device,
                    dtype=hist_pos.dtype,
                ),
            ],
            dim=1,
        )
        context_head_vector = torch.stack([context_heading.cos(), context_heading.sin()], dim=-1)
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=context_pos,
            head_a=context_heading,
            head_vector_a=context_head_vector,
            num_agent=context_pos.shape[0],
            mask=context_mask_full,
        )
        for layer in range(self.num_layers):
            context_feat = self.context_t_attn_layers[layer](
                context_feat.reshape(-1, self.hidden_dim),
                r_t,
                edge_index_t,
            ).reshape(context_feat.shape[0], context_feat.shape[1], self.hidden_dim)
        context_flat_mask = context_mask_full.reshape(-1)
        return {
            'feat': context_feat,
            'feat_flat_valid': context_feat.reshape(-1, self.hidden_dim)[context_flat_mask],
            'pos': context_pos,
            'heading': context_heading,
            'head_vector': context_head_vector,
            'mask': context_mask_full,
            'time': context_raw_time,
            'batch_agent': self._agent_batch(data, context_mask),
        }

    def _future_target_mask(self, data: HeteroData, anchor_index: int) -> torch.Tensor:
        """학습용 target agent를 고른다.

        Args:
            data: SMART 입력.
            anchor_index: 현재 raw frame index.

        Returns:
            shape (A,) bool.
        """
        valid_now = data['agent']['valid_mask'][:, anchor_index]
        valid_future = data['agent']['valid_mask'][:, anchor_index:anchor_index + self.future_window_steps + 1].all(dim=1)
        not_background = data['agent']['type'] != 3
        category = data['agent']['category'] == self.target_category
        return valid_now & valid_future & not_background & category

    def _context_mask(self, data: HeteroData, anchor_index: int) -> torch.Tensor:
        """현재 시각에 scene context로 둘 agent를 고른다.

        Args:
            data: SMART 입력.
            anchor_index: 현재 raw frame index.

        Returns:
            shape (A,) bool.
        """
        valid_now = data['agent']['valid_mask'][:, anchor_index]
        not_background = data['agent']['type'] != 3
        return valid_now & not_background

    def _build_future_supervision(
        self,
        data: HeteroData,
        anchor_index: int,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """정답 2.0초 미래를 local 4-segment 표현으로 만든다.

        Args:
            data: SMART 입력.
            anchor_index: 현재 raw frame index.
            target_mask: shape (A,) bool.

        Returns:
            정답 미래 관련 tensor 묶음.
        """
        future_pos_global = data['agent']['position'][
            target_mask,
            anchor_index:anchor_index + self.future_window_steps + 1,
            :2,
        ]
        future_heading_global = data['agent']['heading'][
            target_mask,
            anchor_index:anchor_index + self.future_window_steps + 1,
        ]
        anchor_pos = future_pos_global[:, 0]
        anchor_heading = future_heading_global[:, 0]
        future_local = local_future_from_global(
            positions=future_pos_global,
            headings=future_heading_global,
            anchor_pos=anchor_pos,
            anchor_heading=anchor_heading,
        )
        target_segments = chunk_future_21_to_4x6(future_local)
        return {
            'anchor_pos': anchor_pos,
            'anchor_heading': anchor_heading,
            'future_local': future_local,
            'future_pos_global': future_pos_global,
            'future_heading_global': future_heading_global,
            'target_segments': target_segments,
        }

    def _build_future_query_feature(
        self,
        data: HeteroData,
        target_mask: torch.Tensor,
        segment_state: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        """noisy 미래 조각을 query token으로 바꾼다.

        Args:
            data: SMART 입력.
            target_mask: shape (A,) bool.
            segment_state: shape (A_tgt, 4, 6, 4).
            flow_time: shape (A_tgt, 1).

        Returns:
            shape (A_tgt, 4, 128).
        """
        num_target = segment_state.shape[0]
        seg_feat = self.future_segment_emb(segment_state.reshape(num_target * self.future_segments, -1))
        seg_feat = seg_feat.reshape(num_target, self.future_segments, self.hidden_dim)
        time_feat = self.flow_time_emb(
            continuous_inputs=flow_time.repeat(1, self.future_segments).reshape(-1, 1),
            categorical_embs=None,
        ).reshape(num_target, self.future_segments, self.hidden_dim)
        seg_ids = torch.arange(self.future_segments, device=segment_state.device).view(1, self.future_segments)
        seg_ids = seg_ids.repeat(num_target, 1)
        seg_idx_feat = self.segment_index_emb(seg_ids)
        if data['agent']['shape'].dim() == 3:
            shape = data['agent']['shape'][target_mask, self.num_historical_steps - 1, :]
        else:
            shape = data['agent']['shape'][target_mask]
        if shape.shape[-1] < 3:
            pad = torch.zeros(shape.shape[0], 3 - shape.shape[-1], device=shape.device, dtype=shape.dtype)
            shape = torch.cat([shape, pad], dim=-1)
        type_feat = self.type_a_emb(data['agent']['type'][target_mask].long()).unsqueeze(1).repeat(1, self.future_segments, 1)
        shape_feat = self.shape_emb(shape).unsqueeze(1).repeat(1, self.future_segments, 1)
        return seg_feat + time_feat + seg_idx_feat + type_feat + shape_feat

    def _build_future_representative_pose(
        self,
        segment_state: torch.Tensor,
        anchor_pos: torch.Tensor,
        anchor_heading: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """각 미래 조각의 대표 global pose를 만든다.

        Args:
            segment_state: shape (A_tgt, 4, 6, 4).
            anchor_pos: shape (A_tgt, 2).
            anchor_heading: shape (A_tgt,).

        Returns:
            rep_pos: shape (A_tgt, 4, 2)
            rep_heading: shape (A_tgt, 4)
            rep_time: shape (A_tgt, 4)
        """
        num_target = segment_state.shape[0]
        flat_segment = segment_state.reshape(num_target * self.future_segments, self.segment_points, self.state_dim)
        flat_anchor_pos = anchor_pos.unsqueeze(1).repeat(1, self.future_segments, 1).reshape(-1, 2)
        flat_anchor_heading = anchor_heading.unsqueeze(1).repeat(1, self.future_segments).reshape(-1)
        rep_pos, rep_heading = global_last_pose_from_local_segment(
            segment=flat_segment,
            anchor_pos=flat_anchor_pos,
            anchor_heading=flat_anchor_heading,
        )
        rep_pos = rep_pos.reshape(num_target, self.future_segments, 2)
        rep_heading = rep_heading.reshape(num_target, self.future_segments)
        rep_time = torch.tensor(
            [5.0, 10.0, 15.0, 20.0],
            device=segment_state.device,
            dtype=segment_state.dtype,
        ).view(1, self.future_segments).repeat(num_target, 1)
        return rep_pos, rep_heading, rep_time

    def _build_history_to_future_edge(
        self,
        context_enc: Dict[str, torch.Tensor],
        future_pos: torch.Tensor,
        future_heading: torch.Tensor,
        future_time: torch.Tensor,
        future_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """scene-wide history memory에서 미래 query로 가는 sparse edge를 만든다.

        Args:
            context_enc: `_encode_context()` 결과.
            future_pos: shape (A_tgt, 4, 2).
            future_heading: shape (A_tgt, 4).
            future_time: shape (A_tgt, 4).
            future_batch: shape (A_tgt,).

        Returns:
            edge_index: shape (2, E)
            relation_emb: shape (E, 128)
        """
        context_mask = context_enc['mask'].reshape(-1)
        context_pos = context_enc['pos'].reshape(-1, self.input_dim)[context_mask]
        context_heading = context_enc['heading'].reshape(-1)[context_mask]
        context_head_vector = context_enc['head_vector'].reshape(-1, 2)[context_mask]
        context_time = context_enc['time'].reshape(-1)[context_mask]
        context_batch = context_enc['batch_agent'].unsqueeze(1).repeat(1, self.history_slots + self.current_slot).reshape(-1)[context_mask]

        future_pos_flat = future_pos.reshape(-1, self.input_dim)
        future_heading_flat = future_heading.reshape(-1)
        future_head_vector = torch.stack([future_heading_flat.cos(), future_heading_flat.sin()], dim=-1)
        future_time_flat = future_time.reshape(-1)
        future_batch_flat = future_batch.unsqueeze(1).repeat(1, self.future_segments).reshape(-1)

        edge_index = radius(
            x=future_pos_flat[:, :2],
            y=context_pos[:, :2],
            r=self.a2a_radius,
            batch_x=future_batch_flat,
            batch_y=context_batch,
            max_num_neighbors=300,
        )
        rel_pos = context_pos[edge_index[0]] - future_pos_flat[edge_index[1]]
        rel_heading = wrap_angle(context_heading[edge_index[0]] - future_heading_flat[edge_index[1]])
        rel_time = (future_time_flat[edge_index[1]] - context_time[edge_index[0]]) * 0.1
        relation = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=future_head_vector[edge_index[1]],
                    nbr_vector=rel_pos[:, :2],
                ),
                rel_heading,
                rel_time,
            ],
            dim=-1,
        )
        relation = self.r_hist_emb(continuous_inputs=relation, categorical_embs=None)
        return edge_index, relation

    def _build_future_sparse_edges(
        self,
        data: HeteroData,
        target_mask: torch.Tensor,
        future_pos: torch.Tensor,
        future_heading: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """future temporal / map / future-future edge를 한 번에 만든다.

        Args:
            data: SMART 입력.
            target_mask: shape (A,) bool.
            future_pos: shape (A_tgt, 4, 2)
            future_heading: shape (A_tgt, 4)

        Returns:
            edge tensor 묶음.
        """
        num_target, num_step, _ = future_pos.shape
        future_head_vector = torch.stack([future_heading.cos(), future_heading.sin()], dim=-1)
        future_mask = torch.ones(num_target, num_step, dtype=torch.bool, device=future_pos.device)
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=future_pos,
            head_a=future_heading,
            head_vector_a=future_head_vector,
            num_agent=num_target,
            mask=future_mask,
        )

        if isinstance(data, Batch):
            batch_agent = data['agent']['batch'][target_mask]
            batch_s = torch.cat([batch_agent + data.num_graphs * t for t in range(num_step)], dim=0)
            batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t for t in range(num_step)], dim=0)
        else:
            batch_agent = torch.zeros(num_target, dtype=torch.long, device=future_pos.device)
            batch_s = torch.arange(num_step, device=future_pos.device).repeat_interleave(num_target)
            batch_pl = torch.arange(num_step, device=future_pos.device).repeat_interleave(data['pt_token']['num_nodes'])

        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=future_pos,
            head_a=future_heading,
            head_vector_a=future_head_vector,
            batch_s=batch_s,
            mask_s=future_mask.transpose(0, 1).reshape(-1),
        )
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            data=data,
            num_step=num_step,
            agent_category=data['agent']['category'][target_mask],
            pos_a=future_pos,
            head_a=future_heading,
            head_vector_a=future_head_vector,
            mask=future_mask,
            batch_s=batch_s,
            batch_pl=batch_pl,
        )
        return {
            'edge_index_t': edge_index_t,
            'r_t': r_t,
            'edge_index_a2a': edge_index_a2a,
            'r_a2a': r_a2a,
            'edge_index_pl2a': edge_index_pl2a,
            'r_pl2a': r_pl2a,
            'batch_agent': batch_agent,
        }

    def _repeat_map_memory(self, x_pt: torch.Tensor, num_step: int) -> torch.Tensor:
        """map memory를 time-major shape로 반복한다.

        Args:
            x_pt: shape (P, 128)
            num_step: 반복할 미래 조각 수.

        Returns:
            shape (P * num_step, 128)
        """
        return (
            x_pt.repeat_interleave(repeats=num_step, dim=0)
            .reshape(-1, num_step, self.hidden_dim)
            .transpose(0, 1)
            .reshape(-1, self.hidden_dim)
        )

    def _predict_velocity(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        context_enc: Dict[str, torch.Tensor],
        target_mask: torch.Tensor,
        anchor_index: int,
        anchor_pos: torch.Tensor,
        anchor_heading: torch.Tensor,
        segment_state: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        """현재 noisy future state에서 flow velocity를 예측한다.

        Args:
            data: SMART 입력.
            map_enc: SMART map encoder 출력.
            context_enc: `_encode_context()` 결과.
            target_mask: shape (A,) bool.
            anchor_index: 현재 raw frame index. 현재 구현에서는 설명용 인자다.
            anchor_pos: shape (A_tgt, 2)
            anchor_heading: shape (A_tgt,)
            segment_state: shape (A_tgt, 4, 6, 4)
            flow_time: shape (A_tgt, 1)

        Returns:
            shape (A_tgt, 4, 6, 4)의 velocity field.
        """
        del anchor_index
        future_feat = self._build_future_query_feature(
            data=data,
            target_mask=target_mask,
            segment_state=segment_state,
            flow_time=flow_time,
        )
        rep_pos, rep_heading, rep_time = self._build_future_representative_pose(
            segment_state=segment_state,
            anchor_pos=anchor_pos,
            anchor_heading=anchor_heading,
        )
        sparse_edges = self._build_future_sparse_edges(
            data=data,
            target_mask=target_mask,
            future_pos=rep_pos,
            future_heading=rep_heading,
        )
        edge_index_hist, r_hist = self._build_history_to_future_edge(
            context_enc=context_enc,
            future_pos=rep_pos,
            future_heading=rep_heading,
            future_time=rep_time,
            future_batch=sparse_edges['batch_agent'],
        )
        map_memory = self._repeat_map_memory(map_enc['x_pt'], self.future_segments)
        context_memory = context_enc['feat_flat_valid']

        for layer in range(self.num_layers):
            future_feat = self.future_t_attn_layers[layer](
                future_feat.reshape(-1, self.hidden_dim),
                sparse_edges['r_t'],
                sparse_edges['edge_index_t'],
            ).reshape(future_feat.shape[0], future_feat.shape[1], self.hidden_dim)
            future_flat = self.hist2fut_attn_layers[layer](
                (context_memory, future_feat.reshape(-1, self.hidden_dim)),
                r_hist,
                edge_index_hist,
            )
            future_flat = self.pt2a_attn_layers[layer](
                (map_memory, future_flat),
                sparse_edges['r_pl2a'],
                sparse_edges['edge_index_pl2a'],
            )
            future_flat = self.a2a_attn_layers[layer](
                future_flat,
                sparse_edges['r_a2a'],
                sparse_edges['edge_index_a2a'],
            )
            future_feat = future_flat.reshape(-1, self.future_segments, self.hidden_dim)
        velocity = self.segment_out_head(future_feat).reshape(
            future_feat.shape[0],
            self.future_segments,
            self.segment_points,
            self.state_dim,
        )
        return velocity

    def _forward_single_anchor(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        anchor_index: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """한 anchor에 대한 open-loop flow 학습 출력을 만든다.

        Args:
            data: SMART 입력.
            map_enc: map encoder 출력.
            anchor_index: 현재 raw frame index.

        Returns:
            target가 없으면 None, 있으면 loss 계산용 tensor 묶음.
        """
        context_mask = self._context_mask(data, anchor_index)
        target_mask = self._future_target_mask(data, anchor_index)
        if int(target_mask.sum().item()) == 0:
            return None

        context_enc = self._encode_context(data, anchor_index, context_mask)
        target = self._build_future_supervision(data, anchor_index, target_mask)
        _, x_t, flow_time, flow_gt = build_ot_flow_path(
            target=target['target_segments'],
            eps=self.flow_eps,
        )
        flow_pred = self._predict_velocity(
            data=data,
            map_enc=map_enc,
            context_enc=context_enc,
            target_mask=target_mask,
            anchor_index=anchor_index,
            anchor_pos=target['anchor_pos'],
            anchor_heading=target['anchor_heading'],
            segment_state=x_t,
            flow_time=flow_time,
        )
        clean_pred = x_t + (1.0 - flow_time.view(-1, 1, 1, 1)) * flow_pred
        future_pred = assemble_4x6_to_21(clean_pred)
        overlap_error = overlap_consistency_error(clean_pred)
        ade = torch.norm(
            future_pred[..., :2] - target['future_local'][..., :2],
            p=2,
            dim=-1,
        ).mean(dim=-1)
        return {
            'flow_pred': flow_pred,
            'flow_gt': flow_gt,
            'flow_time': flow_time,
            'clean_pred_segments': clean_pred,
            'target_segments': target['target_segments'],
            'future_pred_local': future_pred,
            'future_gt_local': target['future_local'],
            'overlap_error': overlap_error,
            'open_loop_ade': ade,
            'anchor_index': torch.full(
                (flow_pred.shape[0],),
                int(anchor_index),
                device=flow_pred.device,
                dtype=torch.long,
            ),
        }

    def _cat_anchor_outputs(self, outputs: Tuple[Dict[str, torch.Tensor], ...]) -> Dict[str, torch.Tensor]:
        """여러 anchor 결과를 loss 계산용 하나의 묶음으로 합친다.

        Args:
            outputs: anchor별 결과 tuple.

        Returns:
            각 tensor가 첫 축으로 이어진 dict.
        """
        merged: Dict[str, torch.Tensor] = {}
        for key in outputs[0].keys():
            merged[key] = torch.cat([item[key] for item in outputs], dim=0)
        return merged

    def forward(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """open-loop 학습과 검증에서 쓰는 forward.

        Args:
            data: SMART 입력.
            map_enc: map encoder 출력.

        Returns:
            flow loss 계산에 필요한 tensor 묶음.
        """
        anchors = self._select_anchor_indices(data)
        anchor_outputs = []
        for anchor_index in anchors.tolist():
            one_anchor = self._forward_single_anchor(data, map_enc, int(anchor_index))
            if one_anchor is not None:
                anchor_outputs.append(one_anchor)
        if len(anchor_outputs) == 0:
            empty = torch.zeros(0, device=data['agent']['position'].device)
            return {
                'flow_pred': empty.view(0, self.future_segments, self.segment_points, self.state_dim),
                'flow_gt': empty.view(0, self.future_segments, self.segment_points, self.state_dim),
                'flow_time': empty.view(0, 1),
                'clean_pred_segments': empty.view(0, self.future_segments, self.segment_points, self.state_dim),
                'target_segments': empty.view(0, self.future_segments, self.segment_points, self.state_dim),
                'future_pred_local': empty.view(0, self.future_window_steps + 1, self.state_dim),
                'future_gt_local': empty.view(0, self.future_window_steps + 1, self.state_dim),
                'overlap_error': empty.view(0, self.future_segments - 1),
                'open_loop_ade': empty.view(0),
                'anchor_index': empty.view(0).long(),
            }
        return self._cat_anchor_outputs(tuple(anchor_outputs))

    def _nearest_token_index(self, agent_type: torch.Tensor, local_traj_xy: torch.Tensor) -> torch.Tensor:
        """예측한 0.5초 중심 궤적을 가장 가까운 SMART motion token으로 바꾼다.

        Args:
            agent_type: shape (N,) uint8 또는 long.
            local_traj_xy: shape (N, 6, 2).

        Returns:
            shape (N,) long.
        """
        num_agent = local_traj_xy.shape[0]
        out = torch.zeros(num_agent, dtype=torch.long, device=local_traj_xy.device)
        type_to_key = {0: 'veh', 1: 'ped', 2: 'cyc'}
        for type_id, key in type_to_key.items():
            mask = agent_type == type_id
            if int(mask.sum().item()) == 0:
                continue
            token_traj = torch.from_numpy(self.trajectory_token_traj[key]).to(local_traj_xy.device).to(local_traj_xy.dtype)
            pred = local_traj_xy[mask]
            if token_traj.dim() == 4:
                token_traj = token_traj.mean(dim=2)
            dist = ((pred[:, None, :, :] - token_traj[None, :, :, :]) ** 2).mean(dim=(-1, -2))
            out[mask] = dist.argmin(dim=1)
        return out

    def _local_future_to_global(
        self,
        future_local: torch.Tensor,
        anchor_pos: torch.Tensor,
        anchor_heading: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """local 21-step 미래를 global 좌표로 바꾼다.

        Args:
            future_local: shape (N, 21, 4)
            anchor_pos: shape (N, 2)
            anchor_heading: shape (N,)

        Returns:
            future_pos_global: shape (N, 21, 2)
            future_heading_global: shape (N, 21)
        """
        local_xy = future_local[..., :2]
        cos = anchor_heading.cos()
        sin = anchor_heading.sin()
        rot = torch.zeros(anchor_pos.shape[0], 2, 2, device=anchor_pos.device, dtype=anchor_pos.dtype)
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = sin
        rot[:, 1, 0] = -sin
        rot[:, 1, 1] = cos
        future_pos_global = torch.bmm(local_xy, rot) + anchor_pos.unsqueeze(1)
        delta_heading = torch.atan2(future_local[..., 2], future_local[..., 3])
        future_heading_global = anchor_heading.unsqueeze(1) + delta_heading
        return future_pos_global, future_heading_global

    def _predict_segments_from_noise(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        anchor_index: int,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """추론 때 2.0초 future를 ODE 적분으로 생성한다.

        Args:
            data: 현재 rollout state가 들어 있는 HeteroData.
            map_enc: map encoder 출력.
            anchor_index: 현재 raw frame index.
            target_mask: shape (A,) bool.

        Returns:
            생성된 미래와 중간 tensor 묶음.
        """
        context_mask = self._context_mask(data, anchor_index)
        context_enc = self._encode_context(data, anchor_index, context_mask)
        current = self._build_current_anchor_feature(data, anchor_index, target_mask)
        anchor_pos = current[1]
        anchor_heading = current[2]
        x = torch.randn(
            int(target_mask.sum().item()),
            self.future_segments,
            self.segment_points,
            self.state_dim,
            device=anchor_pos.device,
            dtype=anchor_pos.dtype,
        )
        for step in range(self.ode_steps):
            t = torch.full(
                (x.shape[0], 1),
                (step + 0.5) / float(self.ode_steps),
                device=x.device,
                dtype=x.dtype,
            )
            velocity = self._predict_velocity(
                data=data,
                map_enc=map_enc,
                context_enc=context_enc,
                target_mask=target_mask,
                anchor_index=anchor_index,
                anchor_pos=anchor_pos,
                anchor_heading=anchor_heading,
                segment_state=x,
                flow_time=t,
            )
            x = x + velocity / float(self.ode_steps)
        future_local = assemble_4x6_to_21(x)
        future_pos_global, future_heading_global = self._local_future_to_global(
            future_local=future_local,
            anchor_pos=anchor_pos,
            anchor_heading=anchor_heading,
        )
        return {
            'future_local': future_local,
            'future_pos_global': future_pos_global,
            'future_heading_global': future_heading_global,
        }

    def inference(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """8초 closed-loop rollout을 수행한다.

        구현을 복잡하게 만들지 않기 위해, 각 0.5초 step마다 fresh noise에서 2.0초를
        다시 생성한다. warm start는 일부 성능 이득이 있을 수 있지만, 현재 목표인
        최소 수정과 안정성을 위해 기본 경로에서는 넣지 않는다.

        Args:
            data: SMART 입력.
            map_enc: map encoder 출력.

        Returns:
            pred_traj: shape (A, 80, 2)
            pred_head: shape (A, 80)
            gt: shape (A, 80, 2)
            valid_mask: shape (A, 80)
        """
        rollout_data = data.clone()
        total_future = rollout_data['agent']['position'].shape[1] - self.num_historical_steps
        pred_traj = torch.zeros(
            rollout_data['agent']['num_nodes'],
            total_future,
            self.input_dim,
            device=rollout_data['agent']['position'].device,
            dtype=rollout_data['agent']['position'].dtype,
        )
        pred_head = torch.zeros(
            rollout_data['agent']['num_nodes'],
            total_future,
            device=rollout_data['agent']['heading'].device,
            dtype=rollout_data['agent']['heading'].dtype,
        )
        initial_gt = data['agent']['position'][:, self.num_historical_steps:, : self.input_dim].contiguous()
        initial_valid = data['agent']['valid_mask'][:, self.num_historical_steps:].clone()

        rollout_steps = total_future // self.shift
        for rollout_step in range(rollout_steps):
            current_raw_index = self.num_historical_steps - 1 + rollout_step * self.shift
            target_mask = rollout_data['agent']['valid_mask'][:, current_raw_index] & (rollout_data['agent']['type'] != 3)
            if int(target_mask.sum().item()) == 0:
                continue
            pred = self._predict_segments_from_noise(
                data=rollout_data,
                map_enc=map_enc,
                anchor_index=current_raw_index,
                target_mask=target_mask,
            )
            step_pos = pred['future_pos_global'][:, 1 : self.shift + 1]
            step_heading = pred['future_heading_global'][:, 1 : self.shift + 1]
            target_indices = torch.nonzero(target_mask).squeeze(-1)

            rollout_data['agent']['position'][target_indices, current_raw_index + 1: current_raw_index + self.shift + 1, : self.input_dim] = step_pos
            if rollout_data['agent']['position'].shape[-1] > self.input_dim:
                rollout_data['agent']['position'][target_indices, current_raw_index + 1: current_raw_index + self.shift + 1, self.input_dim:] = rollout_data['agent']['position'][target_indices, current_raw_index: current_raw_index + 1, self.input_dim:].repeat(1, self.shift, 1)
            rollout_data['agent']['heading'][target_indices, current_raw_index + 1: current_raw_index + self.shift + 1] = step_heading
            rollout_data['agent']['valid_mask'][target_indices, current_raw_index + 1: current_raw_index + self.shift + 1] = True

            vel = torch.cat(
                [
                    step_pos[:, 0:1] - rollout_data['agent']['position'][target_indices, current_raw_index: current_raw_index + 1, : self.input_dim],
                    step_pos[:, 1:] - step_pos[:, :-1],
                ],
                dim=1,
            ) / 0.1
            if rollout_data['agent']['velocity'].shape[-1] >= 2:
                rollout_data['agent']['velocity'][target_indices, current_raw_index + 1: current_raw_index + self.shift + 1, :2] = vel
            if rollout_data['agent']['velocity'].shape[-1] > 2:
                rollout_data['agent']['velocity'][target_indices, current_raw_index + 1: current_raw_index + self.shift + 1, 2:] = 0.0

            current_token_index = current_raw_index // self.shift - 1
            next_token_index = current_token_index + 1
            next_token_pos = pred['future_pos_global'][:, self.shift]
            next_token_heading = pred['future_heading_global'][:, self.shift]
            local_first_chunk = pred['future_local'][:, 0: self.shift + 1, :2]
            nearest_token = self._nearest_token_index(
                agent_type=rollout_data['agent']['type'][target_mask],
                local_traj_xy=local_first_chunk,
            )
            rollout_data['agent']['token_pos'][target_indices, next_token_index] = next_token_pos
            rollout_data['agent']['token_heading'][target_indices, next_token_index] = next_token_heading
            rollout_data['agent']['token_idx'][target_indices, next_token_index] = nearest_token
            rollout_data['agent']['agent_valid_mask'][target_indices, next_token_index] = True
            token_vel = (next_token_pos - rollout_data['agent']['token_pos'][target_indices, current_token_index]) / (0.1 * self.shift)
            rollout_data['agent']['token_velocity'][target_indices, next_token_index] = token_vel

            pred_traj[target_indices, rollout_step * self.shift: (rollout_step + 1) * self.shift] = step_pos
            pred_head[target_indices, rollout_step * self.shift: (rollout_step + 1) * self.shift] = step_heading

        return {
            'pred_traj': pred_traj,
            'pred_head': pred_head,
            'gt': initial_gt,
            'valid_mask': initial_valid,
        }
