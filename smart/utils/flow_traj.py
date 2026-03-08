from __future__ import annotations

from typing import Tuple

import torch


def get_valid_anchor_indices(
    total_steps: int,
    num_historical_steps: int,
    future_window_steps: int,
    shift: int,
    device: torch.device,
) -> torch.Tensor:
    """생성 anchor로 쓸 수 있는 raw 시각 목록을 만든다.

    이 구현은 SMART의 0.5초 token 간격을 그대로 따르기 위해, 현재 시각이
    반드시 `shift`의 배수 간격에 놓이도록 잡는다. 또한 2.0초 정답 구간이
    끝까지 남아 있는 anchor만 남긴다.

    Args:
        total_steps: scene 전체 raw 시점 수. shape 기준으로는 `agent.position.shape[1]`.
        num_historical_steps: SMART 설정의 과거 raw 길이.
        future_window_steps: 한 번에 생성할 미래 raw 길이. 현재 설계에서는 20.
        shift: SMART token 간격. 현재 구현에서는 5.
        device: 결과 tensor를 둘 장치.

    Returns:
        shape (A,)의 정수 tensor. 각 값은 raw frame index다.
    """
    start = num_historical_steps - 1
    end = total_steps - future_window_steps
    if end <= start:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.arange(start, end, shift, dtype=torch.long, device=device)


def chunk_future_21_to_4x6(future: torch.Tensor) -> torch.Tensor:
    """21개 시점 미래를 4개의 겹치는 0.5초 조각으로 바꾼다.

    Args:
        future: shape (..., 21, 4).
            마지막 차원은 `(x_local, y_local, sin(dyaw), cos(dyaw))` 순서를 쓴다.

    Returns:
        shape (..., 4, 6, 4).
            각 조각은 `[0:6], [5:11], [10:16], [15:21]`을 사용한다.
    """
    segments = [
        future[..., 0:6, :],
        future[..., 5:11, :],
        future[..., 10:16, :],
        future[..., 15:21, :],
    ]
    return torch.stack(segments, dim=-3)


def assemble_4x6_to_21(segments: torch.Tensor) -> torch.Tensor:
    """겹치는 4개 조각을 다시 21개 미래 시점으로 합친다.

    겹치는 경계점은 단순 평균으로 합친다. 구조가 단순하고, 추가 튜닝 항이
    생기지 않도록 하기 위해 이 방식을 쓴다.

    Args:
        segments: shape (..., 4, 6, 4).

    Returns:
        shape (..., 21, 4).
    """
    out_shape = list(segments.shape[:-3]) + [21, segments.shape[-1]]
    out = torch.zeros(*out_shape, device=segments.device, dtype=segments.dtype)
    cnt = torch.zeros(*out_shape[:-1], 1, device=segments.device, dtype=segments.dtype)
    starts = [0, 5, 10, 15]
    for seg_idx, start in enumerate(starts):
        out[..., start:start + 6, :] += segments[..., seg_idx, :, :]
        cnt[..., start:start + 6, :] += 1
    return out / cnt.clamp_min(1.0)


def overlap_consistency_error(segments: torch.Tensor) -> torch.Tensor:
    """이웃 조각 사이 경계가 얼마나 어긋나는지 계산한다.

    Args:
        segments: shape (N, 4, 6, 4).

    Returns:
        shape (N, 3)의 제곱 오차 평균값.
    """
    first = segments[:, :-1, -1, :]
    second = segments[:, 1:, 0, :]
    return ((first - second) ** 2).mean(dim=-1)


def build_ot_flow_path(target: torch.Tensor, eps: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """선형 conditional flow matching 학습용 경로를 만든다.

    여기서는 가장 단순한 선형 경로를 사용한다.

    * noise ~ N(0, I)
    * x_t = (1 - t) * noise + t * target
    * u_t = target - noise

    Args:
        target: shape (N, 4, 6, 4)의 정답 미래 조각.
        eps: t가 0에 너무 가깝지 않게 막는 작은 값.

    Returns:
        noise: shape (N, 4, 6, 4)
        x_t: shape (N, 4, 6, 4)
        t: shape (N, 1)
        u_t: shape (N, 4, 6, 4)
    """
    noise = torch.randn_like(target)
    t = torch.rand(target.shape[0], 1, device=target.device, dtype=target.dtype)
    t = eps + (1.0 - eps) * t
    t_view = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_view) * noise + t_view * target
    u_t = target - noise
    return noise, x_t, t, u_t


def build_straight_warm_start(
    current_state: torch.Tensor,
    num_segments: int,
    segment_points: int,
    dt: float,
) -> torch.Tensor:
    """현재 속도를 그대로 유지하는 아주 단순한 초기 미래를 만든다.

    이 함수는 추론 warm start가 꼭 필요할 때만 쓰기 위한 보조 함수다. 지금
    기본 추론에서는 매 step fresh noise를 쓰므로, 코드 경로가 단순하게 유지된다.

    Args:
        current_state: shape (N, 8).
            순서는 `(vx_local, vy_local, sin(yaw), cos(yaw), yaw_rate, length, width, type)`.
        num_segments: 미래 조각 수. 현재 설계에서는 4.
        segment_points: 조각당 점 개수. 현재 설계에서는 6.
        dt: raw frame 간격 초 단위. Waymo 기준 0.1.

    Returns:
        shape (N, 4, 6, 4) tensor.
    """
    num_agents = current_state.shape[0]
    vx = current_state[:, 0:1]
    vy = current_state[:, 1:2]
    yaw_rate = current_state[:, 4:5]
    starts = [0, 5, 10, 15]
    segments = []
    for start in starts[:num_segments]:
        step_ids = torch.arange(start, start + segment_points, device=current_state.device, dtype=current_state.dtype)
        step_ids = step_ids.view(1, -1, 1) * dt
        x = vx.unsqueeze(1) * step_ids
        y = vy.unsqueeze(1) * step_ids
        dyaw = yaw_rate.unsqueeze(1) * step_ids
        seg = torch.cat([x, y, dyaw.sin(), dyaw.cos()], dim=-1)
        segments.append(seg)
    return torch.stack(segments, dim=1)


def global_last_pose_from_local_segment(
    segment: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_heading: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """local 조각의 마지막 점을 global 좌표의 대표 pose로 바꾼다.

    Args:
        segment: shape (N, 6, 4).
        anchor_pos: shape (N, 2).
        anchor_heading: shape (N,).

    Returns:
        last_pos_global: shape (N, 2)
        last_heading_global: shape (N,)
    """
    last_xy = segment[:, -1, :2]
    cos = anchor_heading.cos()
    sin = anchor_heading.sin()
    rot = torch.zeros(anchor_pos.shape[0], 2, 2, device=anchor_pos.device, dtype=anchor_pos.dtype)
    rot[:, 0, 0] = cos
    rot[:, 0, 1] = sin
    rot[:, 1, 0] = -sin
    rot[:, 1, 1] = cos
    last_global = torch.bmm(last_xy.unsqueeze(1), rot).squeeze(1) + anchor_pos
    delta_heading = torch.atan2(segment[:, -1, 2], segment[:, -1, 3])
    return last_global, anchor_heading + delta_heading


def local_future_from_global(
    positions: torch.Tensor,
    headings: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_heading: torch.Tensor,
) -> torch.Tensor:
    """global 미래 궤적을 agent-local 좌표계로 바꾼다.

    Args:
        positions: shape (N, 21, 2).
        headings: shape (N, 21).
        anchor_pos: shape (N, 2).
        anchor_heading: shape (N,).

    Returns:
        shape (N, 21, 4).
    """
    rel = positions - anchor_pos.unsqueeze(1)
    cos = anchor_heading.cos()
    sin = anchor_heading.sin()
    rot = torch.zeros(anchor_pos.shape[0], 2, 2, device=anchor_pos.device, dtype=anchor_pos.dtype)
    rot[:, 0, 0] = cos
    rot[:, 0, 1] = -sin
    rot[:, 1, 0] = sin
    rot[:, 1, 1] = cos
    local_xy = torch.bmm(rel, rot)
    delta_heading = headings - anchor_heading.unsqueeze(1)
    return torch.stack(
        [
            local_xy[..., 0],
            local_xy[..., 1],
            delta_heading.sin(),
            delta_heading.cos(),
        ],
        dim=-1,
    )
