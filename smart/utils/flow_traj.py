import math
from typing import Optional

import torch

from smart.utils.geometry import wrap_angle


def normalize_heading_components(states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    states = states.clone()
    heading = states[..., 2:4]
    norm = torch.linalg.norm(heading, dim=-1, keepdim=True).clamp_min(eps)
    states[..., 2:4] = heading / norm
    return states


def trajectory_to_local_frame(positions: torch.Tensor,
                              headings: torch.Tensor) -> torch.Tensor:
    anchor_pos = positions[:, :1]
    anchor_heading = headings[:, 0]
    cos = anchor_heading.cos()
    sin = anchor_heading.sin()
    rot = positions.new_zeros(positions.size(0), 2, 2)
    rot[:, 0, 0] = cos
    rot[:, 0, 1] = -sin
    rot[:, 1, 0] = sin
    rot[:, 1, 1] = cos
    local_xy = torch.bmm(positions - anchor_pos, rot)
    delta_heading = wrap_angle(headings - anchor_heading.unsqueeze(-1))
    local = positions.new_zeros(positions.size(0), positions.size(1), 4)
    local[..., :2] = local_xy
    local[..., 2] = delta_heading.sin()
    local[..., 3] = delta_heading.cos()
    return local


def local_to_global_future(local_future: torch.Tensor,
                           anchor_pos: torch.Tensor,
                           anchor_heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = anchor_heading.cos()
    sin = anchor_heading.sin()
    rot = local_future.new_zeros(local_future.size(0), 2, 2)
    rot[:, 0, 0] = cos
    rot[:, 0, 1] = sin
    rot[:, 1, 0] = -sin
    rot[:, 1, 1] = cos
    global_xy = torch.bmm(local_future[..., :2], rot) + anchor_pos.unsqueeze(1)
    delta_heading = torch.atan2(local_future[..., 2], local_future[..., 3])
    global_heading = wrap_angle(delta_heading + anchor_heading.unsqueeze(-1))
    return global_xy, global_heading


def chunk_future_21_to_4x6(future: torch.Tensor,
                           action_len: int = 6,
                           action_overlap: int = 1) -> torch.Tensor:
    step = action_len - action_overlap
    chunks = []
    start = 0
    while start + action_len <= future.size(1):
        chunks.append(future[:, start:start + action_len])
        start += step
    return torch.stack(chunks, dim=1)


def assemble_4x6_to_21(chunks: torch.Tensor) -> torch.Tensor:
    num_nodes, num_chunks, action_len, feat_dim = chunks.shape
    total_steps = (num_chunks - 1) * (action_len - 1) + action_len
    future = chunks.new_zeros(num_nodes, total_steps, feat_dim)
    counts = chunks.new_zeros(num_nodes, total_steps, 1)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * (action_len - 1)
        future[:, start:start + action_len] += chunks[:, chunk_idx]
        counts[:, start:start + action_len] += 1
    return normalize_heading_components(future / counts.clamp_min(1))


def build_ot_flow_path(target: torch.Tensor,
                       t: torch.Tensor,
                       noise: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(target)
    t_view = t.view(-1, 1, 1, 1)
    noised = (1.0 - t_view) * noise + t_view * target
    return normalize_heading_components(noised), noise


def midpoint_ode(x_init: torch.Tensor,
                 model_fn,
                 num_steps: int,
                 eps: float = 1e-3) -> torch.Tensor:
    x = x_init
    time_grid = torch.linspace(eps, 1.0, steps=num_steps + 1, device=x.device, dtype=x.dtype)
    for step_idx in range(num_steps):
        t0 = time_grid[step_idx]
        t1 = time_grid[step_idx + 1]
        h = t1 - t0

        t_tensor = torch.full((x.size(0),), t0, device=x.device, dtype=x.dtype)
        x_start = normalize_heading_components(model_fn(x, t_tensor))
        v0 = (x_start - x) / (1.0 - t0).clamp_min(eps)
        x_mid = x + 0.5 * h * v0

        t_mid = t0 + 0.5 * h
        t_mid_tensor = torch.full((x.size(0),), t_mid, device=x.device, dtype=x.dtype)
        x_start_mid = normalize_heading_components(model_fn(x_mid, t_mid_tensor))
        v_mid = (x_start_mid - x_mid) / (1.0 - t_mid).clamp_min(eps)
        x = x + h * v_mid

    return normalize_heading_components(x)


def warm_start_from_previous(previous_xy: torch.Tensor,
                             previous_heading: torch.Tensor,
                             future_window_steps: int = 20,
                             shift: int = 5) -> torch.Tensor:
    tail_xy = previous_xy[:, shift:]
    tail_heading = previous_heading[:, shift:]
    extra_steps = future_window_steps + 1 - tail_xy.size(1)
    if extra_steps > 0:
        velocity = tail_xy[:, -1] - tail_xy[:, -2]
        yaw_rate = wrap_angle(tail_heading[:, -1] - tail_heading[:, -2])
        extra_xy = []
        extra_heading = []
        last_xy = tail_xy[:, -1]
        last_heading = tail_heading[:, -1]
        for _ in range(extra_steps):
            last_xy = last_xy + velocity
            last_heading = wrap_angle(last_heading + yaw_rate)
            extra_xy.append(last_xy)
            extra_heading.append(last_heading)
        tail_xy = torch.cat([tail_xy, torch.stack(extra_xy, dim=1)], dim=1)
        tail_heading = torch.cat([tail_heading, torch.stack(extra_heading, dim=1)], dim=1)
    local_future = trajectory_to_local_frame(tail_xy[:, :future_window_steps + 1],
                                             tail_heading[:, :future_window_steps + 1])
    return chunk_future_21_to_4x6(local_future)
