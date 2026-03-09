from __future__ import annotations

from typing import Callable, Tuple

import torch

from smart.utils.geometry import wrap_angle


def get_valid_anchor_indices(
    total_steps: int,
    num_historical_steps: int,
    future_window_steps: int,
    shift: int,
    device: torch.device,
) -> torch.Tensor:
    """생성 anchor로 쓸 수 있는 raw 시각 목록을 만든다.

    Args:
        total_steps: scene 전체 raw 시점 수.
        num_historical_steps: 과거 raw 길이.
        future_window_steps: 한 번에 생성할 미래 raw 길이.
        shift: SMART token 간격.
        device: 결과 tensor를 둘 장치.

    Returns:
        torch.Tensor: shape `(K,)` 정수 tensor.
            각 값은 현재 anchor의 raw frame index다.
    """
    start = num_historical_steps - 1
    end = total_steps - future_window_steps
    if end <= start:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.arange(start, end, shift, dtype=torch.long, device=device)


def normalize_heading_components(states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """마지막 두 값의 크기를 1로 맞춘다.

    마지막 두 값은 `(sin, cos)`를 뜻한다고 가정한다.

    Args:
        states: shape `(..., 4)` tensor.
            마지막 차원 순서는 `(x, y, sin, cos)`다.
        eps: 0으로 나누는 일을 막는 작은 값.

    Returns:
        torch.Tensor: 입력과 같은 shape.
            마지막 두 값만 정규화된 tensor.
    """
    heading = states[..., 2:4]
    norm = torch.linalg.norm(heading, dim=-1, keepdim=True).clamp_min(eps)
    normalized_heading = heading / norm
    return torch.cat([states[..., :2], normalized_heading], dim=-1)


def chunk_future_21_to_4x6(future: torch.Tensor) -> torch.Tensor:
    """21개 시점 미래를 4개의 겹치는 0.5초 조각으로 바꾼다.

    Args:
        future: shape `(..., 21, 4)`.
            마지막 차원은 `(x_local, y_local, sin(dyaw), cos(dyaw))` 순서다.

    Returns:
        torch.Tensor: shape `(..., 4, 6, 4)`.
            각 조각은 `[0:6]`, `[5:11]`, `[10:16]`, `[15:21]`을 사용한다.
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

    겹치는 경계점은 평균으로 합친 뒤, `(sin, cos)` 쌍의 크기를 1로 맞춘다.

    Args:
        segments: shape `(..., 4, 6, 4)`.

    Returns:
        torch.Tensor: shape `(..., 21, 4)`.
    """
    out_shape = list(segments.shape[:-3]) + [21, segments.shape[-1]]
    out = torch.zeros(*out_shape, device=segments.device, dtype=segments.dtype)
    cnt = torch.zeros(*out_shape[:-1], 1, device=segments.device, dtype=segments.dtype)
    starts = [0, 5, 10, 15]
    for seg_idx, start in enumerate(starts):
        out[..., start:start + 6, :] += segments[..., seg_idx, :, :]
        cnt[..., start:start + 6, :] += 1
    out = out / cnt.clamp_min(1.0)
    return normalize_heading_components(out)


def overlap_consistency_error(segments: torch.Tensor) -> torch.Tensor:
    """이웃 조각 사이 경계가 얼마나 어긋나는지 계산한다.

    Args:
        segments: shape `(N, 4, 6, 4)`.

    Returns:
        torch.Tensor: shape `(N, 3)`.
            각 이웃 조각 경계의 제곱 오차 평균값.
    """
    first = segments[:, :-1, -1, :]
    second = segments[:, 1:, 0, :]
    return ((first - second) ** 2).mean(dim=-1)


def build_ot_flow_path(
    target: torch.Tensor,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """선형 conditional flow matching 학습용 경로를 만든다.

    Args:
        target: shape `(N, 4, 6, 4)`의 정답 미래 조각.
        eps: `t`가 0에 너무 가까워지지 않게 막는 작은 값.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - noise: shape `(N, 4, 6, 4)`
            - x_t: shape `(N, 4, 6, 4)`
            - t: shape `(N, 1)`
            - u_t: shape `(N, 4, 6, 4)`
    """
    noise = torch.randn_like(target)
    t = torch.rand(target.shape[0], 1, device=target.device, dtype=target.dtype)
    t = eps + (1.0 - eps) * t
    t_view = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_view) * noise + t_view * target
    u_t = target - noise
    return noise, x_t, t, u_t


def midpoint_ode_solve(
    x_init: torch.Tensor,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    steps: int,
    normalize_heading: bool = True,
) -> torch.Tensor:
    """2차 midpoint 적분으로 flow ODE를 푼다.

    Args:
        x_init: shape `(N, 4, 6, 4)` 초기 상태.
        velocity_fn: `(x, t) -> v` 함수.
            `x`, `v`의 shape은 모두 `(N, 4, 6, 4)`다.
            `t`의 shape은 `(N, 1)`이다.
        steps: 적분 step 수.
        normalize_heading: True면 각 step 뒤에 `(sin, cos)`를 정규화한다.

    Returns:
        torch.Tensor: shape `(N, 4, 6, 4)`.
            적분이 끝난 최종 상태.
    """
    if steps <= 0:
        raise ValueError('steps must be positive')
    x = x_init
    h = 1.0 / float(steps)
    for step_idx in range(steps):
        t0 = torch.full(
            (x.shape[0], 1),
            float(step_idx) * h,
            device=x.device,
            dtype=x.dtype,
        )
        v0 = velocity_fn(x, t0)
        x_mid = x + 0.5 * h * v0
        if normalize_heading:
            x_mid = normalize_heading_components(x_mid)
        t_mid = torch.full(
            (x.shape[0], 1),
            (float(step_idx) + 0.5) * h,
            device=x.device,
            dtype=x.dtype,
        )
        v_mid = velocity_fn(x_mid, t_mid)
        x = x + h * v_mid
        if normalize_heading:
            x = normalize_heading_components(x)
    return x


def global_last_pose_from_local_segment(
    segment: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_heading: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """local 조각의 마지막 점을 global 좌표의 대표 pose로 바꾼다.

    Args:
        segment: shape `(N, 6, 4)`.
        anchor_pos: shape `(N, 2)`.
        anchor_heading: shape `(N,)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - last_pos_global: shape `(N, 2)`
            - last_heading_global: shape `(N,)`
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
    return last_global, wrap_angle(anchor_heading + delta_heading)


def local_future_from_global(
    positions: torch.Tensor,
    headings: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_heading: torch.Tensor,
) -> torch.Tensor:
    """global 미래 궤적을 agent-local 좌표계로 바꾼다.

    Args:
        positions: shape `(N, 21, 2)`.
        headings: shape `(N, 21)`.
        anchor_pos: shape `(N, 2)`.
        anchor_heading: shape `(N,)`.

    Returns:
        torch.Tensor: shape `(N, 21, 4)`.
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
    delta_heading = wrap_angle(headings - anchor_heading.unsqueeze(1))
    out = torch.stack(
        [
            local_xy[..., 0],
            local_xy[..., 1],
            delta_heading.sin(),
            delta_heading.cos(),
        ],
        dim=-1,
    )
    return normalize_heading_components(out)
