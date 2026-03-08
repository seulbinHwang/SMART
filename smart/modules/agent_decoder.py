from __future__ import annotations

import math
import pickle
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph

from smart.layers import MLPLayer
from smart.layers.attention_layer import AttentionLayer
from smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
from smart.utils.flow_traj import assemble_4x6_to_21, chunk_future_21_to_4x6, midpoint_ode_solve


def cal_polygon_contour(x: float, y: float, theta: float, width: float, length: float) -> List[Tuple[float, float]]:
    """차량 중심과 크기에서 4개 모서리 좌표를 만든다.

    Args:
        x: 중심 x 좌표.
        y: 중심 y 좌표.
        theta: heading.
        width: 폭.
        length: 길이.

    Returns:
        4개 모서리 좌표 리스트.
    """
    left_front_x = x + 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_front_y = y + 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)

    right_front_x = x + 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_front_y = y + 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)

    right_back_x = x - 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_back_y = y - 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)

    left_back_x = x - 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_back_y = y - 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    return [
        (left_front_x, left_front_y),
        (right_front_x, right_front_y),
        (right_back_x, right_back_y),
        (left_back_x, left_back_y),
    ]


class SMARTAgentDecoder(nn.Module):
    """SMART의 agent NTP head를 flow-matching head로 바꾼 decoder.

    이 구현은 기존 SMART의 아래 요소를 그대로 재사용한다.
    1. type / shape embedding
    2. temporal / map-to-agent / agent-agent sparse attention
    3. token library 기반 history state space

    바뀌는 것은 미래 생성부뿐이다.
    미래는 2.0초를 4개의 0.5초 segment로 나눈 연속값으로 생성한다.

    Notes:
        - 현재 구현은 공식 SMART 공개 설정과 같은 batch_size=1 사용을 전제로 한다.
        - shape 표기는 아래와 같다.
          - X_map: [P, 128]
          - H_hist: [A, T_hist<=6, 128]
          - noisy future segments: [A, 4, 6, 4]
          - clean future segments: [A, 4, 6, 4]
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
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        self.shift = 5
        self.future_window_steps = 20
        self.future_segments = 4
        self.segment_points = 6
        self.max_hist_tokens = self.time_span // self.shift
        self.ode_steps = 4
        self.token_size = token_size
        self.hist_mask = False

        self.type_a_emb = nn.Embedding(4, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(input_dim=2, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_hist_emb = FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2a_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.token_emb_veh = MLPEmbedding(input_dim=8, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=8, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=8, hidden_dim=hidden_dim)
        self.fusion_emb = MLPEmbedding(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)

        self.current_anchor_emb = MLPEmbedding(input_dim=8, hidden_dim=hidden_dim)
        self.future_segment_emb = MLPEmbedding(input_dim=24, hidden_dim=hidden_dim)
        self.flow_time_emb = MLPEmbedding(input_dim=1, hidden_dim=hidden_dim)
        self.segment_out_head = MLPLayer(hidden_dim, hidden_dim, 24)

        self.t_attn_layers = nn.ModuleList(
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
        self.hist_attn_layers = nn.ModuleList(
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
        self.pt2a_attn_layers = nn.ModuleList(
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
        self.a2a_attn_layers = nn.ModuleList(
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

        self.trajectory_token = token_data["token"]
        self.trajectory_token_all = token_data["token_all"]
        self.apply(weight_init)

        self._cached_token_state_lib: Dict[str, torch.Tensor] = {}

    # ---------------------------------------------------------------------
    # token / state helpers
    # ---------------------------------------------------------------------
    def _build_token_state_library(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """토큰 라이브러리를 local state 시퀀스로 바꾼다.

        Returns:
            dict[str, Tensor].
            각 type별 shape는 [K_vocab, 6, 4].
            마지막 차원 4는 [x_local, y_local, sin(dyaw), cos(dyaw)] 이다.
        """
        if self._cached_token_state_lib:
            return {k: v.to(device) for k, v in self._cached_token_state_lib.items()}

        out: Dict[str, torch.Tensor] = {}
        for k in ["veh", "ped", "cyc"]:
            tok_last = torch.from_numpy(self.trajectory_token[k]).float()  # [K, 4, 2]
            tok_hist = torch.from_numpy(self.trajectory_token_all[k]).float()[:, : self.shift]  # [K, 5, 4, 2]
            tok = torch.cat([tok_hist, tok_last[:, None]], dim=1)  # [K, 6, 4, 2]
            center = tok.mean(dim=2)  # [K, 6, 2]
            diff = tok[:, :, 0] - tok[:, :, 3]
            heading = torch.atan2(diff[..., 1], diff[..., 0])
            state = torch.stack(
                [center[..., 0], center[..., 1], torch.sin(heading), torch.cos(heading)],
                dim=-1,
            )
            out[k] = state
        self._cached_token_state_lib = out
        return {k: v.to(device) for k, v in out.items()}

    def _nearest_token_index(self, seg_local: torch.Tensor, agent_type: torch.Tensor) -> torch.Tensor:
        """첫 0.5초 segment를 SMART token 하나로 다시 바꾼다.

        Args:
            seg_local: shape [A, 6, 4].
            agent_type: shape [A]. 0=veh, 1=ped, 2=cyc.

        Returns:
            shape [A]. nearest token index.
        """
        lib = self._build_token_state_library(seg_local.device)
        out = torch.zeros(seg_local.size(0), device=seg_local.device, dtype=torch.long)
        masks = {
            "veh": agent_type == 0,
            "ped": agent_type == 1,
            "cyc": agent_type == 2,
        }
        for k, m in masks.items():
            if not torch.any(m):
                continue
            d = (seg_local[m][:, None] - lib[k][None]).pow(2).mean(dim=(-1, -2))
            out[m] = torch.argmin(d, dim=1)
        return out

    def _agent_token_embedding(
        self,
        data: HeteroData,
        token_pos: torch.Tensor,
        token_heading: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """기존 SMART token history embedding을 그대로 만든다.

        Args:
            token_pos: shape [A, T_slot, 2].
            token_heading: shape [A, T_slot].
            token_idx: shape [A, T_slot].

        Returns:
            shape [A, T_slot, 128].
        """
        num_agent, num_step, _ = token_pos.shape
        motion_vector = torch.cat(
            [
                token_pos.new_zeros(num_agent, 1, self.input_dim),
                token_pos[:, 1:] - token_pos[:, :-1],
            ],
            dim=1,
        )
        head_vector = torch.stack([token_heading.cos(), token_heading.sin()], dim=-1)
        agent_type = data["agent"]["type"]

        tok_veh = torch.from_numpy(self.trajectory_token["veh"]).to(token_pos.device).float()
        tok_ped = torch.from_numpy(self.trajectory_token["ped"]).to(token_pos.device).float()
        tok_cyc = torch.from_numpy(self.trajectory_token["cyc"]).to(token_pos.device).float()
        emb_veh = self.token_emb_veh(tok_veh.view(tok_veh.size(0), -1))
        emb_ped = self.token_emb_ped(tok_ped.view(tok_ped.size(0), -1))
        emb_cyc = self.token_emb_cyc(tok_cyc.view(tok_cyc.size(0), -1))

        tok_emb = torch.zeros(num_agent, num_step, self.hidden_dim, device=token_pos.device)
        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2
        tok_emb[veh_mask] = emb_veh[token_idx[veh_mask]]
        tok_emb[ped_mask] = emb_ped[token_idx[ped_mask]]
        tok_emb[cyc_mask] = emb_cyc[token_idx[cyc_mask]]

        categorical_embs = [
            self.type_a_emb(agent_type.long()).repeat_interleave(repeats=num_step, dim=0),
            self.shape_emb(data["agent"]["shape"][:, self.num_historical_steps - 1, :]).repeat_interleave(
                repeats=num_step,
                dim=0,
            ),
        ]
        feat = torch.stack(
            [
                torch.norm(motion_vector[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector, nbr_vector=motion_vector[:, :, :2]),
            ],
            dim=-1,
        )
        x_a = self.x_a_emb(feat.view(-1, 2), categorical_embs=categorical_embs).view(num_agent, num_step, self.hidden_dim)
        return self.fusion_emb(torch.cat([tok_emb, x_a], dim=-1))

    def _build_current_anchor_token(
        self,
        data: HeteroData,
        cur_pos_world: torch.Tensor,
        cur_heading_world: torch.Tensor,
        cur_pos_prev_world: torch.Tensor,
        cur_heading_prev_world: torch.Tensor,
    ) -> torch.Tensor:
        """현재 정확한 상태를 128차원 anchor token으로 만든다.

        Args:
            cur_pos_world: shape [A, 2].
            cur_heading_world: shape [A].
            cur_pos_prev_world: shape [A, 2].
            cur_heading_prev_world: shape [A].

        Returns:
            shape [A, 128].
        """
        vel_world = (cur_pos_world - cur_pos_prev_world) / 0.1
        cos_h = torch.cos(cur_heading_world)
        sin_h = torch.sin(cur_heading_world)
        v_x_local = vel_world[:, 0] * cos_h + vel_world[:, 1] * sin_h
        v_y_local = -vel_world[:, 0] * sin_h + vel_world[:, 1] * cos_h
        yaw_rate = wrap_angle(cur_heading_world - cur_heading_prev_world) / 0.1
        shape = data["agent"]["shape"][:, self.num_historical_steps - 1, :2]
        type_id = data["agent"]["type"].float().unsqueeze(-1)
        anchor_in = torch.cat(
            [
                v_x_local.unsqueeze(-1),
                v_y_local.unsqueeze(-1),
                torch.sin(cur_heading_world).unsqueeze(-1),
                torch.cos(cur_heading_world).unsqueeze(-1),
                yaw_rate.unsqueeze(-1),
                shape,
                type_id,
            ],
            dim=-1,
        )
        return self.current_anchor_emb(anchor_in)

    def _extract_history_window(
        self,
        hist_feat_full: torch.Tensor,
        token_pos: torch.Tensor,
        token_heading: torch.Tensor,
        current_slot: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """최근 최대 6개 history token window를 만든다.

        Args:
            hist_feat_full: shape [A, T_slot, 128].
            token_pos: shape [A, T_slot, 2].
            token_heading: shape [A, T_slot].
            current_slot: 현재 slot index.

        Returns:
            hist_feat: [A, 6, 128]
            hist_pos: [A, 6, 2]
            hist_heading: [A, 6]
            hist_mask: [A, 6]
        """
        a, _, h = hist_feat_full.shape
        hist_feat = hist_feat_full.new_zeros(a, self.max_hist_tokens, h)
        hist_pos = token_pos.new_zeros(a, self.max_hist_tokens, 2)
        hist_heading = token_heading.new_zeros(a, self.max_hist_tokens)
        hist_mask = torch.zeros(a, self.max_hist_tokens, device=hist_feat_full.device, dtype=torch.bool)

        start = max(0, current_slot - self.max_hist_tokens + 1)
        src_feat = hist_feat_full[:, start : current_slot + 1]
        src_pos = token_pos[:, start : current_slot + 1]
        src_head = token_heading[:, start : current_slot + 1]
        l = src_feat.size(1)
        hist_feat[:, -l:] = src_feat
        hist_pos[:, -l:] = src_pos
        hist_heading[:, -l:] = src_head
        hist_mask[:, -l:] = True
        return hist_feat, hist_pos, hist_heading, hist_mask

    # ---------------------------------------------------------------------
    # geometry helpers
    # ---------------------------------------------------------------------
    def _local_segment_last_pose(self, segments_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """segment 마지막 점의 local pose를 뽑는다.

        Args:
            segments_local: shape [A, 4, 6, 4].

        Returns:
            last_xy: [A, 4, 2]
            last_heading: [A, 4]
        """
        last_xy = segments_local[:, :, -1, :2]
        last_heading = torch.atan2(segments_local[:, :, -1, 2], segments_local[:, :, -1, 3])
        return last_xy, last_heading

    def _local_to_world(
        self,
        cur_pos_world: torch.Tensor,
        cur_heading_world: torch.Tensor,
        local_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """local state sequence를 world 중심점 / heading으로 바꾼다.

        Args:
            cur_pos_world: shape [A, 2].
            cur_heading_world: shape [A].
            local_states: shape [A, T, 4].

        Returns:
            world_xy: [A, T, 2]
            world_heading: [A, T]
        """
        cos_h = torch.cos(cur_heading_world)[:, None]
        sin_h = torch.sin(cur_heading_world)[:, None]
        x = local_states[..., 0]
        y = local_states[..., 1]
        xw = x * cos_h - y * sin_h + cur_pos_world[:, None, 0]
        yw = x * sin_h + y * cos_h + cur_pos_world[:, None, 1]
        hw = wrap_angle(torch.atan2(local_states[..., 2], local_states[..., 3]) + cur_heading_world[:, None])
        return torch.stack([xw, yw], dim=-1), hw

    def _world_to_local(
        self,
        cur_pos_world: torch.Tensor,
        cur_heading_world: torch.Tensor,
        world_xy: torch.Tensor,
        world_heading: torch.Tensor,
    ) -> torch.Tensor:
        """world 중심점 / heading을 current anchor 기준 local state로 바꾼다.

        Args:
            cur_pos_world: shape [A, 2].
            cur_heading_world: shape [A].
            world_xy: shape [A, T, 2].
            world_heading: shape [A, T].

        Returns:
            shape [A, T, 4].
        """
        rel = world_xy - cur_pos_world[:, None]
        cos_h = torch.cos(cur_heading_world)[:, None]
        sin_h = torch.sin(cur_heading_world)[:, None]
        xl = rel[..., 0] * cos_h + rel[..., 1] * sin_h
        yl = -rel[..., 0] * sin_h + rel[..., 1] * cos_h
        dhead = wrap_angle(world_heading - cur_heading_world[:, None])
        return torch.stack([xl, yl, torch.sin(dhead), torch.cos(dhead)], dim=-1)

    def _segment_to_box_corners(self, seg_local: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        """local segment state를 box corner trajectory로 바꾼다.

        Args:
            seg_local: shape [A, 6, 4].
            shape: shape [A, 2]. [length, width]

        Returns:
            shape [A, 6, 4, 2].
        """
        a = seg_local.size(0)
        out = seg_local.new_zeros(a, 6, 4, 2)
        for i in range(a):
            length = float(shape[i, 0].item())
            width = float(shape[i, 1].item())
            for t in range(6):
                x = float(seg_local[i, t, 0].item())
                y = float(seg_local[i, t, 1].item())
                theta = float(torch.atan2(seg_local[i, t, 2], seg_local[i, t, 3]).item())
                contour = cal_polygon_contour(x, y, theta, width, length)
                out[i, t] = seg_local.new_tensor(contour)
        return out

    # ---------------------------------------------------------------------
    # sparse edge builders for future segments
    # ---------------------------------------------------------------------
    def _build_future_temporal_edge(
        self,
        seg_last_xy: torch.Tensor,
        seg_last_heading: torch.Tensor,
        valid_agent_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """같은 agent 안 4개 future segment의 self-attention edge를 만든다.

        Args:
            seg_last_xy: [A, 4, 2]
            seg_last_heading: [A, 4]
            valid_agent_mask: [A]

        Returns:
            edge_index: [2, E]
            rel_emb: [E, 128]
        """
        a, s, _ = seg_last_xy.shape
        head_vec = torch.stack([torch.cos(seg_last_heading), torch.sin(seg_last_heading)], dim=-1)
        rows: List[int] = []
        cols: List[int] = []
        rels: List[torch.Tensor] = []
        for agent_idx in range(a):
            if not bool(valid_agent_mask[agent_idx]):
                continue
            for src in range(s):
                for dst in range(s):
                    if src == dst:
                        continue
                    src_idx = agent_idx * s + src
                    dst_idx = agent_idx * s + dst
                    rel_pos = seg_last_xy[agent_idx, src] - seg_last_xy[agent_idx, dst]
                    rel_head = wrap_angle(seg_last_heading[agent_idx, src] - seg_last_heading[agent_idx, dst])
                    rel = torch.stack(
                        [
                            torch.norm(rel_pos[:2], p=2, dim=-1),
                            angle_between_2d_vectors(head_vec[agent_idx, dst], rel_pos[:2]),
                            rel_head,
                            seg_last_xy.new_tensor(float(src - dst)),
                        ]
                    )
                    rows.append(src_idx)
                    cols.append(dst_idx)
                    rels.append(rel)
        if len(rows) == 0:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=seg_last_xy.device)
            rel_emb = seg_last_xy.new_zeros(0, self.hidden_dim)
            return edge_index, rel_emb
        edge_index = torch.stack(
            [torch.tensor(rows, device=seg_last_xy.device), torch.tensor(cols, device=seg_last_xy.device)], dim=0
        )
        rel = torch.stack(rels, dim=0)
        rel_emb = self.r_t_emb(rel, categorical_embs=None)
        return edge_index, rel_emb

    def _build_future_history_edge(
        self,
        seg_last_xy_local: torch.Tensor,
        seg_last_heading_local: torch.Tensor,
        hist_pos_world: torch.Tensor,
        hist_heading_world: torch.Tensor,
        hist_mask: torch.Tensor,
        cur_pos_world: torch.Tensor,
        cur_heading_world: torch.Tensor,
        valid_agent_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """future query와 history key/value 사이 bipartite edge를 만든다.

        Returns:
            edge_index: [2, E]
                source는 history flat index, target은 future flat index이다.
            rel_emb: [E, 128]
            hist_feat_mask_flat: [A*6]
                history 쪽의 유효 여부를 나타낸다.
        """
        a = seg_last_xy_local.size(0)
        hist_local = self._world_to_local(cur_pos_world, cur_heading_world, hist_pos_world, hist_heading_world)
        hist_xy_local = hist_local[..., :2]
        hist_heading_local = torch.atan2(hist_local[..., 2], hist_local[..., 3])
        future_head_vec = torch.stack([torch.cos(seg_last_heading_local), torch.sin(seg_last_heading_local)], dim=-1)

        rows: List[int] = []
        cols: List[int] = []
        rels: List[torch.Tensor] = []
        hist_flat_mask = hist_mask.reshape(-1)
        for agent_idx in range(a):
            if not bool(valid_agent_mask[agent_idx]):
                continue
            for seg_idx in range(self.future_segments):
                tgt_idx = agent_idx * self.future_segments + seg_idx
                for h_idx in range(self.max_hist_tokens):
                    if not bool(hist_mask[agent_idx, h_idx]):
                        continue
                    src_idx = agent_idx * self.max_hist_tokens + h_idx
                    rel_pos = hist_xy_local[agent_idx, h_idx] - seg_last_xy_local[agent_idx, seg_idx]
                    rel_head = wrap_angle(hist_heading_local[agent_idx, h_idx] - seg_last_heading_local[agent_idx, seg_idx])
                    rel = torch.stack(
                        [
                            torch.norm(rel_pos[:2], p=2, dim=-1),
                            angle_between_2d_vectors(future_head_vec[agent_idx, seg_idx], rel_pos[:2]),
                            rel_head,
                            hist_xy_local.new_tensor(float(h_idx - seg_idx)),
                        ]
                    )
                    rows.append(src_idx)
                    cols.append(tgt_idx)
                    rels.append(rel)
        if len(rows) == 0:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=seg_last_xy_local.device)
            rel_emb = seg_last_xy_local.new_zeros(0, self.hidden_dim)
            return edge_index, rel_emb, hist_flat_mask
        edge_index = torch.stack(
            [torch.tensor(rows, device=seg_last_xy_local.device), torch.tensor(cols, device=seg_last_xy_local.device)], dim=0
        )
        rel_emb = self.r_hist_emb(torch.stack(rels, dim=0), categorical_embs=None)
        return edge_index, rel_emb, hist_flat_mask

    def _build_future_map_edge(
        self,
        data: HeteroData,
        seg_last_xy_local: torch.Tensor,
        seg_last_heading_local: torch.Tensor,
        cur_pos_world: torch.Tensor,
        cur_heading_world: torch.Tensor,
        valid_agent_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """future segment와 map token 사이 sparse edge를 만든다.

        Args:
            seg_last_xy_local: [A, 4, 2]
            seg_last_heading_local: [A, 4]

        Returns:
            edge_index: [2, E]. source는 map index, target은 future flat index.
            rel_emb: [E, 128]
        """
        a = seg_last_xy_local.size(0)
        seg_world_xy, seg_world_heading = self._local_to_world(
            cur_pos_world,
            cur_heading_world,
            torch.cat([seg_last_xy_local, torch.sin(seg_last_heading_local)[..., None], torch.cos(seg_last_heading_local)[..., None]], dim=-1),
        )
        seg_world_xy = seg_world_xy.reshape(a * self.future_segments, 2)
        seg_world_heading = seg_world_heading.reshape(a * self.future_segments)
        future_mask = valid_agent_mask[:, None].repeat(1, self.future_segments).reshape(-1)
        head_vec = torch.stack([torch.cos(seg_world_heading), torch.sin(seg_world_heading)], dim=-1)

        pos_pl = data["pt_token"]["position"][:, : self.input_dim].contiguous()
        orient_pl = data["pt_token"]["orientation"].contiguous()

        if isinstance(data, Batch):
            raise NotImplementedError("이 구현은 공식 공개 설정과 같은 batch_size=1을 전제로 한다.")
        edge_index = radius(
            x=seg_world_xy[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=None,
            batch_y=None,
            max_num_neighbors=300,
        )
        edge_index = edge_index[:, future_mask[edge_index[1]]]
        if edge_index.numel() == 0:
            return edge_index, seg_world_xy.new_zeros(0, self.hidden_dim)
        rel_pos = pos_pl[edge_index[0]] - seg_world_xy[edge_index[1]]
        rel_orient = wrap_angle(orient_pl[edge_index[0]] - seg_world_heading[edge_index[1]])
        rel = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(head_vec[edge_index[1]], rel_pos[:, :2]),
                rel_orient,
            ],
            dim=-1,
        )
        rel_emb = self.r_pt2a_emb(rel, categorical_embs=None)
        return edge_index, rel_emb

    def _build_future_a2a_edge(
        self,
        seg_last_xy_local: torch.Tensor,
        seg_last_heading_local: torch.Tensor,
        cur_pos_world: torch.Tensor,
        cur_heading_world: torch.Tensor,
        valid_agent_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """같은 future index를 가진 agent들끼리의 interaction edge를 만든다."""
        a = seg_last_xy_local.size(0)
        seg_world_xy, seg_world_heading = self._local_to_world(
            cur_pos_world,
            cur_heading_world,
            torch.cat([seg_last_xy_local, torch.sin(seg_last_heading_local)[..., None], torch.cos(seg_last_heading_local)[..., None]], dim=-1),
        )
        pos_s = seg_world_xy.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = seg_world_heading.transpose(0, 1).reshape(-1)
        head_vec_s = torch.stack([torch.cos(head_s), torch.sin(head_s)], dim=-1)
        mask_s = valid_agent_mask[:, None].repeat(1, self.future_segments).transpose(0, 1).reshape(-1)
        batch_s = torch.arange(self.future_segments, device=pos_s.device).repeat_interleave(a)
        edge_index = radius_graph(pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False, max_num_neighbors=300)
        edge_index = subgraph(subset=mask_s, edge_index=edge_index)[0]
        if edge_index.numel() == 0:
            return edge_index, pos_s.new_zeros(0, self.hidden_dim)
        rel_pos = pos_s[edge_index[0]] - pos_s[edge_index[1]]
        rel_head = wrap_angle(head_s[edge_index[0]] - head_s[edge_index[1]])
        rel = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(head_vec_s[edge_index[1]], rel_pos[:, :2]),
                rel_head,
            ],
            dim=-1,
        )
        rel_emb = self.r_a2a_emb(rel, categorical_embs=None)
        return edge_index, rel_emb

    # ---------------------------------------------------------------------
    # core predictor
    # ---------------------------------------------------------------------
    def _predict_clean_segments_from_state(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        token_pos: torch.Tensor,
        token_heading: torch.Tensor,
        token_idx: torch.Tensor,
        current_slot: int,
        cur_pos_world: torch.Tensor,
        cur_heading_world: torch.Tensor,
        cur_pos_prev_world: torch.Tensor,
        cur_heading_prev_world: torch.Tensor,
        noisy_segments: torch.Tensor,
        flow_t: torch.Tensor,
        valid_agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        """현재 rollout state에서 clean future segment를 예측한다.

        Args:
            noisy_segments: shape [A, 4, 6, 4].
            flow_t: shape [A].

        Returns:
            shape [A, 4, 6, 4].
        """
        a = token_pos.size(0)
        hist_feat_full = self._agent_token_embedding(data, token_pos, token_heading, token_idx)
        hist_feat, hist_pos, hist_heading, hist_mask = self._extract_history_window(
            hist_feat_full, token_pos, token_heading, current_slot
        )
        cur_anchor = self._build_current_anchor_token(
            data,
            cur_pos_world,
            cur_heading_world,
            cur_pos_prev_world,
            cur_heading_prev_world,
        )
        future_feat = self.future_segment_emb(noisy_segments.reshape(a, self.future_segments, -1))
        future_feat = future_feat + self.flow_time_emb(flow_t[:, None]).unsqueeze(1)
        future_feat = future_feat + cur_anchor[:, None]
        future_feat = future_feat + self.type_a_emb(data["agent"]["type"].long())[:, None]
        future_feat = future_feat + self.shape_emb(data["agent"]["shape"][:, self.num_historical_steps - 1, :])[:, None]

        seg_last_xy_local, seg_last_heading_local = self._local_segment_last_pose(noisy_segments)
        edge_t, r_t = self._build_future_temporal_edge(seg_last_xy_local, seg_last_heading_local, valid_agent_mask)
        edge_hist, r_hist, hist_flat_mask = self._build_future_history_edge(
            seg_last_xy_local,
            seg_last_heading_local,
            hist_pos,
            hist_heading,
            hist_mask,
            cur_pos_world,
            cur_heading_world,
            valid_agent_mask,
        )
        edge_map, r_map = self._build_future_map_edge(
            data,
            seg_last_xy_local,
            seg_last_heading_local,
            cur_pos_world,
            cur_heading_world,
            valid_agent_mask,
        )
        edge_a2a, r_a2a = self._build_future_a2a_edge(
            seg_last_xy_local,
            seg_last_heading_local,
            cur_pos_world,
            cur_heading_world,
            valid_agent_mask,
        )

        future_feat = future_feat.reshape(-1, self.hidden_dim)
        hist_feat_flat = hist_feat.reshape(-1, self.hidden_dim)
        map_feat = map_enc["x_pt"]

        for i in range(self.num_layers):
            future_feat = self.t_attn_layers[i](future_feat, r_t, edge_t)
            future_feat = self.hist_attn_layers[i]((hist_feat_flat, future_feat), r_hist, edge_hist)
            future_feat = self.pt2a_attn_layers[i]((map_feat, future_feat), r_map, edge_map)
            future_feat = self.a2a_attn_layers[i](future_feat, r_a2a, edge_a2a)

        clean = self.segment_out_head(future_feat).view(a, self.future_segments, self.segment_points, 4)
        # 현재점과 공유 경계점은 직접 고정한다.
        clean[:, 0, 0, 0] = 0.0
        clean[:, 0, 0, 1] = 0.0
        clean[:, 0, 0, 2] = 0.0
        clean[:, 0, 0, 3] = 1.0
        clean[:, 1:, 0] = clean[:, :-1, -1]
        return clean

    def _vector_field(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        state: Dict[str, torch.Tensor],
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """flow matching vector field를 계산한다."""
        clean = self._predict_clean_segments_from_state(
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
            noisy_segments=z,
            flow_t=t,
            valid_agent_mask=state["valid_now_mask"],
        )
        return (clean - z) / (1.0 - t[:, None, None, None] + 1e-4)

    # ---------------------------------------------------------------------
    # public forward / inference
    # ---------------------------------------------------------------------
    def forward(self, data: HeteroData, map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """open-loop 학습용 forward.

        data 안에는 smart.py가 미리 넣어 둔 아래 field가 있어야 한다.
        - flow_anchor_raw: int
        - flow_anchor_slot: int
        - flow_noisy_segments: [A, 4, 6, 4]
        - flow_t: [A]
        - flow_target_mask: [A]
        """
        if isinstance(data, Batch) and data.num_graphs > 1:
            raise NotImplementedError("현재 구현은 공개 설정과 같은 batch_size=1을 전제로 한다.")

        anchor_raw = int(data["agent"]["flow_anchor_raw"].item())
        anchor_slot = int(data["agent"]["flow_anchor_slot"].item())
        token_pos = data["agent"]["token_pos"].clone()
        token_heading = data["agent"]["token_heading"].clone()
        token_idx = data["agent"]["token_idx"].clone()
        cur_pos_world = data["agent"]["position"][:, anchor_raw, : self.input_dim].contiguous()
        cur_heading_world = data["agent"]["heading"][:, anchor_raw].contiguous()
        prev_raw = max(anchor_raw - 1, 0)
        cur_pos_prev_world = data["agent"]["position"][:, prev_raw, : self.input_dim].contiguous()
        cur_heading_prev_world = data["agent"]["heading"][:, prev_raw].contiguous()
        valid_now_mask = data["agent"]["valid_mask"][:, anchor_raw]

        clean = self._predict_clean_segments_from_state(
            data=data,
            map_enc=map_enc,
            token_pos=token_pos,
            token_heading=token_heading,
            token_idx=token_idx,
            current_slot=anchor_slot,
            cur_pos_world=cur_pos_world,
            cur_heading_world=cur_heading_world,
            cur_pos_prev_world=cur_pos_prev_world,
            cur_heading_prev_world=cur_heading_prev_world,
            noisy_segments=data["agent"]["flow_noisy_segments"],
            flow_t=data["agent"]["flow_t"],
            valid_agent_mask=valid_now_mask,
        )
        return {
            "flow_pred_segments": clean,
            "flow_pred_future": assemble_4x6_to_21(clean),
            "flow_target_mask": data["agent"]["flow_target_mask"],
        }

    def build_initial_rollout_state(self, data: HeteroData, anchor_raw: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """rollout 시작 state를 만든다.

        Args:
            data: SMART batch data.
            anchor_raw: 시작 raw step. None이면 관측 끝 시점(10)을 쓴다.

        Returns:
            rollout state dict.
        """
        if anchor_raw is None:
            anchor_raw = self.num_historical_steps - 1
        anchor_slot = anchor_raw // self.shift
        return {
            "token_pos": data["agent"]["token_pos"].clone(),
            "token_heading": data["agent"]["token_heading"].clone(),
            "token_idx": data["agent"]["token_idx"].clone(),
            "current_raw": torch.tensor(anchor_raw, device=data["agent"]["position"].device),
            "current_slot": torch.tensor(anchor_slot, device=data["agent"]["position"].device),
            "current_pos_world": data["agent"]["position"][:, anchor_raw, : self.input_dim].contiguous().clone(),
            "current_heading_world": data["agent"]["heading"][:, anchor_raw].contiguous().clone(),
            "prev_pos_world": data["agent"]["position"][:, max(anchor_raw - 1, 0), : self.input_dim].contiguous().clone(),
            "prev_heading_world": data["agent"]["heading"][:, max(anchor_raw - 1, 0)].contiguous().clone(),
            "valid_now_mask": data["agent"]["valid_mask"][:, anchor_raw].clone(),
            "prev_future_world_xy": None,
            "prev_future_world_heading": None,
        }

    def _warm_start_noise(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """이전 step 예측 결과를 사용해 다음 2.0초 초기값을 만든다.

        복잡한 proposal을 쓰지 않고, 직전 예측의 뒤 1.5초를 새 current frame으로 옮긴 뒤
        마지막 0.5초만 작은 noise로 채운다.
        """
        a = state["current_pos_world"].size(0)
        if state["prev_future_world_xy"] is None:
            return torch.randn(a, self.future_segments, self.segment_points, 4, device=state["current_pos_world"].device)

        prev_xy = state["prev_future_world_xy"][:, 5:]  # [A, 16, 2]
        prev_head = state["prev_future_world_heading"][:, 5:]  # [A, 16]
        last_xy = prev_xy[:, -1]
        prev_xy_last = prev_xy[:, -2]
        vel = last_xy - prev_xy_last
        last_head = prev_head[:, -1]
        prev_head_last = prev_head[:, -2]
        dhead = wrap_angle(last_head - prev_head_last)

        extra_xy = [last_xy[:, None] + vel[:, None] * float(i + 1) for i in range(5)]
        extra_xy = torch.cat(extra_xy, dim=1)
        extra_head = torch.stack([wrap_angle(last_head + dhead * float(i + 1)) for i in range(5)], dim=1)
        world_xy = torch.cat([prev_xy, extra_xy], dim=1)[:, :21]
        world_head = torch.cat([prev_head, extra_head], dim=1)[:, :21]
        local = self._world_to_local(state["current_pos_world"], state["current_heading_world"], world_xy, world_head)
        z0 = chunk_future_21_to_4x6(local)
        return z0 + 0.05 * torch.randn_like(z0)

    def rollout_step(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """한 번의 0.5초 rollout step을 수행한다.

        Returns:
            state: 갱신된 state
            first_world_xy: [A, 6, 2]
            first_world_heading: [A, 6]
            full_future_world_xy: [A, 21, 2]
        """
        z0 = self._warm_start_noise(state)
        a = z0.size(0)

        def vf(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self._vector_field(data, map_enc, state, z, t)

        z_pred = midpoint_ode_solve(z0, vf, steps=self.ode_steps)
        y_pred = assemble_4x6_to_21(z_pred)
        world_xy, world_heading = self._local_to_world(state["current_pos_world"], state["current_heading_world"], y_pred)
        first_local = z_pred[:, 0]
        first_world_xy = world_xy[:, :6]
        first_world_heading = world_heading[:, :6]

        next_token = self._nearest_token_index(first_local, data["agent"]["type"])
        next_slot = int(state["current_slot"].item()) + 1
        state["token_idx"][:, next_slot] = next_token
        state["token_pos"][:, next_slot] = first_world_xy[:, -1]
        state["token_heading"][:, next_slot] = first_world_heading[:, -1]

        state["prev_pos_world"] = state["current_pos_world"]
        state["prev_heading_world"] = state["current_heading_world"]
        state["current_pos_world"] = first_world_xy[:, -1]
        state["current_heading_world"] = first_world_heading[:, -1]
        state["current_raw"] = state["current_raw"] + self.shift
        state["current_slot"] = state["current_slot"] + 1
        current_raw = int(state["current_raw"].item())
        state["valid_now_mask"] = data["agent"]["valid_mask"][:, min(current_raw, data["agent"]["valid_mask"].size(1) - 1)]
        state["prev_future_world_xy"] = world_xy
        state["prev_future_world_heading"] = world_heading
        return state, first_world_xy, first_world_heading, world_xy

    def inference(self, data: HeteroData, map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """8초 rollout 추론을 수행한다."""
        if isinstance(data, Batch) and data.num_graphs > 1:
            raise NotImplementedError("현재 구현은 공개 설정과 같은 batch_size=1을 전제로 한다.")
        state = self.build_initial_rollout_state(data)
        num_roll_steps = data["agent"]["position"].shape[1] - self.num_historical_steps
        num_token_steps = num_roll_steps // self.shift
        a = data["agent"].num_nodes
        pred_traj = torch.zeros(a, num_roll_steps, 2, device=state["current_pos_world"].device)
        pred_head = torch.zeros(a, num_roll_steps, device=state["current_pos_world"].device)
        next_idx_list: List[torch.Tensor] = []

        for t in range(num_token_steps):
            state, first_world_xy, first_world_heading, _ = self.rollout_step(data, map_enc, state)
            pred_traj[:, t * self.shift : (t + 1) * self.shift] = first_world_xy[:, 1:]
            pred_head[:, t * self.shift : (t + 1) * self.shift] = first_world_heading[:, 1:]
            next_idx_list.append(state["token_idx"][:, int(state["current_slot"].item())][:, None])

        agent_valid_mask = data["agent"]["agent_valid_mask"].clone()
        agent_valid_mask[data["agent"]["category"] != 3] = False
        return {
            "pos_a": state["token_pos"],
            "head_a": state["token_heading"],
            "gt": data["agent"]["position"][:, self.num_historical_steps :, : self.input_dim].contiguous(),
            "valid_mask": agent_valid_mask[:, self.num_historical_steps :],
            "pred_traj": pred_traj,
            "pred_head": pred_head,
            "next_token_idx": torch.cat(next_idx_list, dim=1) if len(next_idx_list) > 0 else torch.zeros(a, 0, dtype=torch.long, device=pred_traj.device),
            "next_token_idx_gt": data["agent"]["token_idx"].roll(shifts=-1, dims=1),
            "next_token_eval_mask": data["agent"]["agent_valid_mask"],
        }
