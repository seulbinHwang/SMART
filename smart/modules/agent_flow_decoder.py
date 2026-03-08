import math
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse

from smart.layers import MLPLayer
from smart.layers.attention_layer import AttentionLayer
from smart.layers.fourier_embedding import MLPEmbedding
from smart.modules.agent_decoder import SMARTAgentDecoder
from smart.utils import angle_between_2d_vectors
from smart.utils import assemble_4x6_to_21
from smart.utils import build_ot_flow_path
from smart.utils import chunk_future_21_to_4x6
from smart.utils import local_to_global_future
from smart.utils import midpoint_ode
from smart.utils import normalize_heading_components
from smart.utils import trajectory_to_local_frame
from smart.utils import warm_start_from_previous
from smart.utils import weight_init
from smart.utils import wrap_angle


class FlowTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256, max_period: float = 10.0) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class SMARTAgentFlowDecoder(SMARTAgentDecoder):

    def __init__(self,
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
                 ode_steps: int = 4,
                 anchor_chunk_k: int = 4) -> None:
        super().__init__(dataset=dataset,
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
                         token_size=token_size)
        self.future_window_steps = future_window_steps
        self.ode_steps = ode_steps
        self.anchor_chunk_k = anchor_chunk_k
        self.hist_slots = max(1, self.time_span // self.shift)
        self.num_future_segments = self.future_window_steps // self.shift
        self.segment_steps = self.shift + 1
        self.flow_eps = 1e-3

        self.current_anchor_emb = MLPEmbedding(input_dim=8, hidden_dim=hidden_dim)
        self.future_segment_emb = MLPEmbedding(input_dim=self.segment_steps * 4, hidden_dim=hidden_dim)
        self.flow_timestep_emb = FlowTimeEmbedding(hidden_dim=hidden_dim)
        self.segment_index_emb = nn.Embedding(self.num_future_segments, hidden_dim)

        self.future_t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.hist_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.future_map_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.future_a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.segment_out_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                         output_dim=self.segment_steps * 4)
        nn.init.normal_(self.segment_index_emb.weight, mean=0.0, std=0.02)

        self.token_center_traj = {}
        for key, contour_prefix in self.trajectory_token_all.items():
            contour = torch.from_numpy(
                np.concatenate([contour_prefix[:, :self.shift], self.trajectory_token[key][:, None]], axis=1)
            ).to(torch.float)
            self.token_center_traj[key] = contour.mean(dim=2)

    def _scene_indices(self, data: HeteroData) -> List[torch.Tensor]:
        if isinstance(data, Batch):
            batch = data['agent']['batch']
            num_scenes = int(batch.max().item()) + 1
            return [torch.where(batch == scene_idx)[0] for scene_idx in range(num_scenes)]
        return [torch.arange(data['agent']['num_nodes'], device=data['agent']['token_pos'].device)]

    def _map_indices(self, data: HeteroData) -> List[torch.Tensor]:
        if isinstance(data, Batch):
            batch = data['pt_token']['batch']
            num_scenes = int(batch.max().item()) + 1
            return [torch.where(batch == scene_idx)[0] for scene_idx in range(num_scenes)]
        return [torch.arange(data['pt_token']['num_nodes'], device=data['pt_token']['position'].device)]

    def _build_scene_inputs(self,
                            data: HeteroData,
                            map_enc: Mapping[str, torch.Tensor],
                            agent_index: torch.Tensor,
                            pt_index: torch.Tensor) -> tuple[Dict, Dict]:
        scene_agent = {
            'num_nodes': int(agent_index.numel()),
            'position': data['agent']['position'][agent_index, :, :self.input_dim].clone(),
            'heading': data['agent']['heading'][agent_index].clone(),
            'velocity': data['agent']['velocity'][agent_index, :, :self.input_dim].clone(),
            'valid_mask': data['agent']['valid_mask'][agent_index].clone(),
            'token_pos': data['agent']['token_pos'][agent_index].clone(),
            'token_heading': data['agent']['token_heading'][agent_index].clone(),
            'token_idx': data['agent']['token_idx'][agent_index].clone(),
            'token_velocity': data['agent']['token_velocity'][agent_index].clone(),
            'agent_valid_mask': data['agent']['agent_valid_mask'][agent_index].clone(),
            'type': data['agent']['type'][agent_index].clone(),
            'category': data['agent']['category'][agent_index].clone(),
            'shape': data['agent']['shape'][agent_index].clone(),
        }
        scene_map = {
            'x_pt': map_enc['x_pt'][pt_index],
            'position': data['pt_token']['position'][pt_index, :self.input_dim].clone(),
            'orientation': data['pt_token']['orientation'][pt_index].clone(),
        }
        return scene_agent, scene_map

    def _build_map2node_edge(self,
                             map_pos: torch.Tensor,
                             map_orient: torch.Tensor,
                             node_pos: torch.Tensor,
                             node_head: torch.Tensor,
                             node_head_vector: torch.Tensor,
                             node_batch: Optional[torch.Tensor] = None,
                             map_batch: Optional[torch.Tensor] = None,
                             node_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = radius(x=node_pos[:, :2], y=map_pos[:, :2], r=self.pl2a_radius,
                            batch_x=node_batch, batch_y=map_batch, max_num_neighbors=300)
        if node_mask is not None:
            edge_index = edge_index[:, node_mask[edge_index[1]]]
        rel_pos = map_pos[edge_index[0]] - node_pos[edge_index[1]]
        rel_orient = wrap_angle(map_orient[edge_index[0]] - node_head[edge_index[1]])
        relation = torch.stack([
            torch.norm(rel_pos[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=node_head_vector[edge_index[1]], nbr_vector=rel_pos[:, :2]),
            rel_orient,
        ], dim=-1)
        return edge_index, self.r_pt2a_emb(continuous_inputs=relation, categorical_embs=None)

    def _encode_context(self, scene_agent: Dict, scene_map: Dict) -> torch.Tensor:
        pos_a = scene_agent['token_pos']
        head_a = scene_agent['token_heading']
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        feat_a, _ = self.agent_token_embedding({'agent': scene_agent},
                                               scene_agent['category'],
                                               scene_agent['token_idx'],
                                               pos_a,
                                               head_vector_a)
        num_agent, num_step, _ = feat_a.shape
        mask = scene_agent['agent_valid_mask'].clone()
        edge_index_t, r_t = self.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask)
        batch_s = torch.arange(num_step, device=pos_a.device).repeat_interleave(num_agent)
        batch_pl = torch.arange(num_step, device=pos_a.device).repeat_interleave(scene_map['position'].size(0))
        mask_s = mask.transpose(0, 1).reshape(-1)
        edge_index_a2a, r_a2a = self.build_interaction_edge(pos_a, head_a, head_vector_a, batch_s, mask_s)
        edge_index_pl2a, r_pl2a = self._build_map2node_edge(
            map_pos=scene_map['position'].repeat(num_step, 1),
            map_orient=scene_map['orientation'].repeat(num_step),
            node_pos=pos_a.transpose(0, 1).reshape(-1, self.input_dim),
            node_head=head_a.transpose(0, 1).reshape(-1),
            node_head_vector=head_vector_a.transpose(0, 1).reshape(-1, 2),
            node_batch=batch_s,
            map_batch=batch_pl,
            node_mask=mask_s,
        )

        for layer_idx in range(self.num_layers):
            feat_a = feat_a.reshape(-1, self.hidden_dim)
            feat_a = self.t_attn_layers[layer_idx](feat_a, r_t, edge_index_t)
            feat_a = feat_a.reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            feat_a = self.pt2a_attn_layers[layer_idx]((scene_map['x_pt'].repeat(num_step, 1), feat_a),
                                                      r_pl2a, edge_index_pl2a)
            feat_a = self.a2a_attn_layers[layer_idx](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.reshape(num_step, num_agent, self.hidden_dim).transpose(0, 1).contiguous()

        return feat_a

    def _anchor_times(self, scene_agent: Dict) -> List[int]:
        start = self.num_historical_steps - 1
        stop = scene_agent['position'].size(1) - self.future_window_steps
        return list(range(start, stop, self.shift))

    def _select_anchor_times(self, scene_agent: Dict) -> List[int]:
        anchors = self._anchor_times(scene_agent)
        if self.training and len(anchors) > self.anchor_chunk_k:
            perm = torch.randperm(len(anchors), device=scene_agent['position'].device)[:self.anchor_chunk_k]
            anchors = [anchors[idx] for idx in perm.sort()[0].tolist()]
        return anchors

    def _build_current_anchor_features(self, scene_agent: Dict, anchor_step: int) -> torch.Tensor:
        heading = scene_agent['heading'][:, anchor_step]
        velocity = scene_agent['position'].new_zeros(scene_agent['num_nodes'], 2)
        if anchor_step > 0:
            velocity = (scene_agent['position'][:, anchor_step, :2] - scene_agent['position'][:, anchor_step - 1, :2]) / 0.1
            valid_pair = scene_agent['valid_mask'][:, anchor_step] & scene_agent['valid_mask'][:, anchor_step - 1]
            velocity[~valid_pair] = 0
        cos = heading.cos()
        sin = heading.sin()
        rot = velocity.new_zeros(velocity.size(0), 2, 2)
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = -sin
        rot[:, 1, 0] = sin
        rot[:, 1, 1] = cos
        local_velocity = torch.bmm(velocity.unsqueeze(1), rot).squeeze(1)

        yaw_rate = heading.new_zeros(heading.size(0))
        if anchor_step > 0:
            yaw_rate = wrap_angle(scene_agent['heading'][:, anchor_step] - scene_agent['heading'][:, anchor_step - 1]) / 0.1
            prev_valid = scene_agent['valid_mask'][:, anchor_step - 1]
            yaw_rate[~prev_valid] = 0

        anchor_state = torch.cat([
            local_velocity,
            heading.sin().unsqueeze(-1),
            heading.cos().unsqueeze(-1),
            yaw_rate.unsqueeze(-1),
            scene_agent['shape'][:, anchor_step, :3],
        ], dim=-1)
        return self.current_anchor_emb(anchor_state) + self.type_a_emb(scene_agent['type'].long())

    def _build_history_memory(self,
                              scene_agent: Dict,
                              context_feat: torch.Tensor,
                              anchor_step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_slot = anchor_step // self.shift - 1
        slot_start = max(0, anchor_slot - self.hist_slots + 1)
        slot_ids = torch.arange(slot_start, anchor_slot + 1, device=context_feat.device)

        hist_feat = context_feat[:, slot_ids].reshape(-1, self.hidden_dim)
        hist_pos = scene_agent['token_pos'][:, slot_ids].reshape(-1, self.input_dim)
        hist_head = scene_agent['token_heading'][:, slot_ids].reshape(-1)
        hist_agent_ids = torch.arange(scene_agent['num_nodes'], device=context_feat.device).unsqueeze(1).repeat(1, slot_ids.numel()).reshape(-1)
        hist_time = (anchor_step - ((slot_ids + 1) * self.shift).repeat(scene_agent['num_nodes'])) * 0.1

        now_feat = self._build_current_anchor_features(scene_agent, anchor_step)
        hist_feat = torch.cat([hist_feat, now_feat], dim=0)
        hist_pos = torch.cat([hist_pos, scene_agent['position'][:, anchor_step, :self.input_dim]], dim=0)
        hist_head = torch.cat([hist_head, scene_agent['heading'][:, anchor_step]], dim=0)
        hist_agent_ids = torch.cat([hist_agent_ids, torch.arange(scene_agent['num_nodes'], device=context_feat.device)], dim=0)
        hist_time = torch.cat([hist_time, hist_time.new_zeros(scene_agent['num_nodes'])], dim=0)
        return hist_feat, hist_pos, hist_head, hist_agent_ids, hist_time

    def _prepare_query_inputs(self,
                              scene_agent: Dict,
                              anchor_step: int,
                              target_index: torch.Tensor,
                              noised_segments: torch.Tensor,
                              t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        type_emb = self.type_a_emb(scene_agent['type'][target_index].long())
        shape_emb = self.shape_emb(scene_agent['shape'][target_index, anchor_step, :3])
        query = self.future_segment_emb(noised_segments.reshape(noised_segments.size(0), self.num_future_segments, -1).reshape(-1, self.segment_steps * 4))
        query = query.reshape(-1, self.num_future_segments, self.hidden_dim)
        query = query + type_emb.unsqueeze(1) + shape_emb.unsqueeze(1)
        query = query + self.segment_index_emb.weight.unsqueeze(0)
        query = query + self.flow_timestep_emb(t).unsqueeze(1)
        return query.reshape(-1, self.hidden_dim), torch.arange(self.num_future_segments, device=query.device).repeat(query.size(0))

    def _future_rep_pose(self,
                         anchor_pos: torch.Tensor,
                         anchor_heading: torch.Tensor,
                         segments: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        future_local = assemble_4x6_to_21(segments)
        future_global_xy, future_global_heading = local_to_global_future(future_local, anchor_pos, anchor_heading)
        end_index = torch.tensor([self.shift, self.shift * 2, self.shift * 3, self.shift * 4],
                                 device=segments.device)
        rep_pos = future_global_xy[:, end_index]
        rep_heading = future_global_heading[:, end_index]
        return rep_pos.reshape(-1, self.input_dim), rep_heading.reshape(-1)

    def _build_future_temporal_edge(self,
                                    rep_pos: torch.Tensor,
                                    rep_heading: torch.Tensor,
                                    query_agent_ids: torch.Tensor,
                                    segment_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        same_agent = query_agent_ids.unsqueeze(1) == query_agent_ids.unsqueeze(0)
        same_agent.fill_diagonal_(False)
        edge_index = dense_to_sparse(same_agent)[0]
        head_vector = torch.stack([rep_heading.cos(), rep_heading.sin()], dim=-1)
        rel_pos = rep_pos[edge_index[0]] - rep_pos[edge_index[1]]
        rel_heading = wrap_angle(rep_heading[edge_index[0]] - rep_heading[edge_index[1]])
        rel_segment = (segment_ids[edge_index[0]] - segment_ids[edge_index[1]]).to(rep_pos.dtype)
        relation = torch.stack([
            torch.norm(rel_pos[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector[edge_index[1]], nbr_vector=rel_pos[:, :2]),
            rel_heading,
            rel_segment,
        ], dim=-1)
        return edge_index, self.r_t_emb(continuous_inputs=relation, categorical_embs=None)

    def _build_history_edge(self,
                            rep_pos: torch.Tensor,
                            rep_heading: torch.Tensor,
                            query_agent_ids: torch.Tensor,
                            segment_ids: torch.Tensor,
                            hist_pos: torch.Tensor,
                            hist_heading: torch.Tensor,
                            hist_agent_ids: torch.Tensor,
                            hist_age: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_radius = radius(x=rep_pos[:, :2], y=hist_pos[:, :2], r=self.a2a_radius, max_num_neighbors=256)
        same_agent = dense_to_sparse((query_agent_ids.unsqueeze(1) == hist_agent_ids.unsqueeze(0)).t())[0]
        edge_index = torch.cat([edge_radius, same_agent], dim=1).t().unique(dim=0).t()
        head_vector = torch.stack([rep_heading.cos(), rep_heading.sin()], dim=-1)
        rel_pos = hist_pos[edge_index[0]] - rep_pos[edge_index[1]]
        rel_heading = wrap_angle(hist_heading[edge_index[0]] - rep_heading[edge_index[1]])
        query_time = (segment_ids[edge_index[1]].to(hist_age.dtype) + 1) * (self.shift * 0.1)
        relation = torch.stack([
            torch.norm(rel_pos[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector[edge_index[1]], nbr_vector=rel_pos[:, :2]),
            rel_heading,
            query_time + hist_age[edge_index[0]],
        ], dim=-1)
        return edge_index, self.r_t_emb(continuous_inputs=relation, categorical_embs=None)

    def _build_future_a2a_edge(self,
                               rep_pos: torch.Tensor,
                               rep_heading: torch.Tensor,
                               segment_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = segment_ids
        edge_index = radius_graph(x=rep_pos[:, :2], r=self.a2a_radius, batch=batch, loop=False, max_num_neighbors=300)
        head_vector = torch.stack([rep_heading.cos(), rep_heading.sin()], dim=-1)
        rel_pos = rep_pos[edge_index[0]] - rep_pos[edge_index[1]]
        rel_heading = wrap_angle(rep_heading[edge_index[0]] - rep_heading[edge_index[1]])
        relation = torch.stack([
            torch.norm(rel_pos[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector[edge_index[1]], nbr_vector=rel_pos[:, :2]),
            rel_heading,
        ], dim=-1)
        return edge_index, self.r_a2a_emb(continuous_inputs=relation, categorical_embs=None)

    def _decode_anchor(self,
                       scene_agent: Dict,
                       scene_map: Dict,
                       context_feat: torch.Tensor,
                       anchor_step: int,
                       target_index: torch.Tensor,
                       noised_segments: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        anchor_pos = scene_agent['position'][target_index, anchor_step, :self.input_dim]
        anchor_heading = scene_agent['heading'][target_index, anchor_step]
        hist_feat, hist_pos, hist_head, hist_agent_ids, hist_age = self._build_history_memory(
            scene_agent, context_feat, anchor_step)

        query_feat, segment_ids = self._prepare_query_inputs(scene_agent, anchor_step, target_index, noised_segments, t)
        query_agent_ids = target_index.repeat_interleave(self.num_future_segments)
        rep_pos, rep_heading = self._future_rep_pose(anchor_pos, anchor_heading, noised_segments)
        rep_head_vector = torch.stack([rep_heading.cos(), rep_heading.sin()], dim=-1)

        edge_index_t, r_t = self._build_future_temporal_edge(rep_pos, rep_heading, query_agent_ids, segment_ids)
        edge_index_hist, r_hist = self._build_history_edge(rep_pos, rep_heading, query_agent_ids, segment_ids,
                                                           hist_pos, hist_head, hist_agent_ids, hist_age)
        edge_index_map, r_map = self._build_map2node_edge(scene_map['position'], scene_map['orientation'],
                                                          rep_pos, rep_heading, rep_head_vector)
        edge_index_a2a, r_a2a = self._build_future_a2a_edge(rep_pos, rep_heading, segment_ids)

        for layer_idx in range(self.num_layers):
            query_feat = self.future_t_attn_layers[layer_idx](query_feat, r_t, edge_index_t)
            query_feat = self.hist_attn_layers[layer_idx]((hist_feat, query_feat), r_hist, edge_index_hist)
            query_feat = self.future_map_attn_layers[layer_idx]((scene_map['x_pt'], query_feat), r_map, edge_index_map)
            query_feat = self.future_a2a_attn_layers[layer_idx](query_feat, r_a2a, edge_index_a2a)

        pred = self.segment_out_head(query_feat).reshape(-1, self.num_future_segments, self.segment_steps, 4)
        return normalize_heading_components(pred)

    def _build_anchor_targets(self,
                              scene_agent: Dict,
                              anchor_step: int,
                              target_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = scene_agent['position'][target_index, anchor_step:anchor_step + self.future_window_steps + 1, :self.input_dim]
        heading = scene_agent['heading'][target_index, anchor_step:anchor_step + self.future_window_steps + 1]
        valid = scene_agent['valid_mask'][target_index, anchor_step:anchor_step + self.future_window_steps + 1]
        future = trajectory_to_local_frame(pos, heading)
        return chunk_future_21_to_4x6(future), chunk_future_21_to_4x6(valid.unsqueeze(-1).to(future.dtype)).squeeze(-1).bool()

    def _forward_anchor(self,
                        scene_agent: Dict,
                        scene_map: Dict,
                        context_feat: torch.Tensor,
                        anchor_step: int) -> Optional[Dict[str, torch.Tensor]]:
        target_mask = (scene_agent['category'] == 3) & scene_agent['valid_mask'][:, anchor_step]
        target_index = torch.where(target_mask)[0]
        if target_index.numel() == 0:
            return None

        target_segments, target_valid = self._build_anchor_targets(scene_agent, anchor_step, target_index)
        t = torch.rand(target_index.numel(), device=target_segments.device) * (1.0 - self.flow_eps) + self.flow_eps
        noised_segments, noise = build_ot_flow_path(target_segments, t)
        pred_segments = self._decode_anchor(scene_agent, scene_map, context_feat, anchor_step,
                                            target_index, noised_segments, t)
        return {
            'pred_segments': pred_segments,
            'target_segments': target_segments,
            'target_valid_mask': target_valid,
            'noise_segments': noise,
        }

    def _concat_results(self, results: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not results:
            empty = torch.empty(0)
            return {
                'pred_segments': empty,
                'target_segments': empty,
                'target_valid_mask': empty.bool(),
                'noise_segments': empty,
            }
        return {
            'pred_segments': torch.cat([item['pred_segments'] for item in results], dim=0),
            'target_segments': torch.cat([item['target_segments'] for item in results], dim=0),
            'target_valid_mask': torch.cat([item['target_valid_mask'] for item in results], dim=0),
            'noise_segments': torch.cat([item['noise_segments'] for item in results], dim=0),
        }

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        results: List[Dict[str, torch.Tensor]] = []
        for agent_index, pt_index in zip(self._scene_indices(data), self._map_indices(data)):
            scene_agent, scene_map = self._build_scene_inputs(data, map_enc, agent_index, pt_index)
            context_feat = self._encode_context(scene_agent, scene_map)
            for anchor_step in self._select_anchor_times(scene_agent):
                anchor_result = self._forward_anchor(scene_agent, scene_map, context_feat, anchor_step)
                if anchor_result is not None:
                    results.append(anchor_result)
        return self._concat_results(results)

    def _nearest_token_index(self,
                             local_future: torch.Tensor,
                             agent_type: torch.Tensor) -> torch.Tensor:
        token_idx = torch.zeros(local_future.size(0), dtype=torch.long, device=local_future.device)
        type_map = {0: 'veh', 1: 'ped', 2: 'cyc'}
        for type_id, type_name in type_map.items():
            mask = agent_type == type_id
            if not mask.any():
                continue
            token_center = self.token_center_traj[type_name].to(local_future.device)
            distance = ((local_future[mask, None, :, :2] - token_center.unsqueeze(0)) ** 2).sum(dim=(-1, -2))
            token_idx[mask] = distance.argmin(dim=-1)
        return token_idx

    def inference(self,
                  data: HeteroData,
                  map_enc: Mapping[str, torch.Tensor],
                  rollout_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        rollout_steps = rollout_steps if rollout_steps is not None else data['agent']['position'].shape[1] - self.num_historical_steps
        rollout_chunks = rollout_steps // self.shift

        pred_traj_list = []
        pred_head_list = []
        gt_list = []
        valid_mask_list = []

        for agent_index, pt_index in zip(self._scene_indices(data), self._map_indices(data)):
            scene_agent, scene_map = self._build_scene_inputs(data, map_enc, agent_index, pt_index)

            current_slot = self.num_historical_steps // self.shift - 1
            scene_agent['token_pos'][:, current_slot + 1:] = 0
            scene_agent['token_heading'][:, current_slot + 1:] = 0
            scene_agent['agent_valid_mask'][:, current_slot + 1:] = False
            scene_agent['position'][:, self.num_historical_steps:, :] = 0
            scene_agent['heading'][:, self.num_historical_steps:] = 0

            rollout_target_mask = (scene_agent['type'] != 3) & scene_agent['valid_mask'][:, self.num_historical_steps - 1]
            rollout_agent_ids = torch.where(rollout_target_mask)[0]

            pred_traj = scene_agent['position'].new_zeros(scene_agent['num_nodes'], rollout_chunks * self.shift, self.input_dim)
            pred_head = scene_agent['heading'].new_zeros(scene_agent['num_nodes'], rollout_chunks * self.shift)
            previous_xy = None
            previous_heading = None

            for chunk_idx in range(rollout_chunks):
                anchor_step = self.num_historical_steps - 1 + chunk_idx * self.shift
                active_local = scene_agent['valid_mask'][rollout_agent_ids, anchor_step]
                active_agent_ids = rollout_agent_ids[active_local]
                if active_agent_ids.numel() == 0:
                    continue

                context_feat = self._encode_context(scene_agent, scene_map)
                if previous_xy is not None:
                    x_init = warm_start_from_previous(previous_xy[active_local], previous_heading[active_local],
                                                      future_window_steps=self.future_window_steps, shift=self.shift)
                else:
                    x_init = torch.randn(active_agent_ids.numel(), self.num_future_segments, self.segment_steps, 4,
                                         device=context_feat.device)
                    x_init = normalize_heading_components(x_init)

                def denoise_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    return self._decode_anchor(scene_agent, scene_map, context_feat, anchor_step,
                                               active_agent_ids, x_t, t)

                pred_segments = midpoint_ode(x_init, denoise_fn, num_steps=self.ode_steps, eps=self.flow_eps)
                pred_future_local = assemble_4x6_to_21(pred_segments)

                anchor_pos = scene_agent['position'][active_agent_ids, anchor_step, :self.input_dim]
                anchor_heading = scene_agent['heading'][active_agent_ids, anchor_step]
                pred_future_xy, pred_future_heading = local_to_global_future(pred_future_local, anchor_pos, anchor_heading)

                pred_traj[active_agent_ids, chunk_idx * self.shift:(chunk_idx + 1) * self.shift] = pred_future_xy[:, 1:self.shift + 1]
                pred_head[active_agent_ids, chunk_idx * self.shift:(chunk_idx + 1) * self.shift] = pred_future_heading[:, 1:self.shift + 1]

                valid_steps = scene_agent['valid_mask'][active_agent_ids, anchor_step + 1:anchor_step + self.shift + 1]
                next_pos = scene_agent['position'][active_agent_ids, anchor_step + 1:anchor_step + self.shift + 1]
                next_head = scene_agent['heading'][active_agent_ids, anchor_step + 1:anchor_step + self.shift + 1]
                scene_agent['position'][active_agent_ids, anchor_step + 1:anchor_step + self.shift + 1] = torch.where(
                    valid_steps.unsqueeze(-1),
                    pred_future_xy[:, 1:self.shift + 1],
                    next_pos,
                )
                scene_agent['heading'][active_agent_ids, anchor_step + 1:anchor_step + self.shift + 1] = torch.where(
                    valid_steps,
                    pred_future_heading[:, 1:self.shift + 1],
                    next_head,
                )

                next_slot = anchor_step // self.shift
                next_valid = scene_agent['valid_mask'][active_agent_ids, anchor_step + self.shift]
                next_token_idx = self._nearest_token_index(pred_future_local[:, :self.segment_steps], scene_agent['type'][active_agent_ids])
                if next_valid.any():
                    scene_agent['token_idx'][active_agent_ids[next_valid], next_slot] = next_token_idx[next_valid]
                    scene_agent['token_pos'][active_agent_ids[next_valid], next_slot] = pred_future_xy[next_valid, self.shift]
                    scene_agent['token_heading'][active_agent_ids[next_valid], next_slot] = pred_future_heading[next_valid, self.shift]
                    scene_agent['agent_valid_mask'][active_agent_ids[next_valid], next_slot] = True

                full_previous_xy = scene_agent['position'].new_zeros(rollout_agent_ids.numel(), self.future_window_steps + 1, self.input_dim)
                full_previous_heading = scene_agent['heading'].new_zeros(rollout_agent_ids.numel(), self.future_window_steps + 1)
                full_previous_xy[active_local] = pred_future_xy
                full_previous_heading[active_local] = pred_future_heading
                previous_xy = full_previous_xy
                previous_heading = full_previous_heading

            pred_traj_list.append(pred_traj)
            pred_head_list.append(pred_head)
            gt_list.append(data['agent']['position'][agent_index, self.num_historical_steps:self.num_historical_steps + rollout_chunks * self.shift, :self.input_dim].contiguous())
            valid_mask_list.append(data['agent']['valid_mask'][agent_index, self.num_historical_steps:self.num_historical_steps + rollout_chunks * self.shift].contiguous())

        return {
            'pred_traj': torch.cat(pred_traj_list, dim=0),
            'pred_head': torch.cat(pred_head_list, dim=0),
            'gt': torch.cat(gt_list, dim=0),
            'valid_mask': torch.cat(valid_mask_list, dim=0),
        }
