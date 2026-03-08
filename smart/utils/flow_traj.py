from __future__ import annotations

from typing import Callable, Tuple

import torch


def chunk_future_21_to_4x6(future: torch.Tensor) -> torch.Tensor:
    """21개 점 미래를 4개의 0.5초 segment로 바꾼다.

    Args:
        future: shape [N, 21, 4].
            현재 1점 + 미래 20점으로 이루어진 연속 상태열이다.
            마지막 차원 4는 [x_local, y_local, sin(dyaw), cos(dyaw)] 이다.

    Returns:
        shape [N, 4, 6, 4].
            4개의 segment 각각이 6개 점을 가진다.
            segment 0: points 0..5
            segment 1: points 5..10
            segment 2: points 10..15
            segment 3: points 15..20
    """
    if future.ndim != 3 or future.size(1) != 21 or future.size(2) != 4:
        raise ValueError(f"future must be [N, 21, 4], got {tuple(future.shape)}")

    segments = [
        future[:, 0:6],
        future[:, 5:11],
        future[:, 10:16],
        future[:, 15:21],
    ]
    return torch.stack(segments, dim=1)


def assemble_4x6_to_21(segments: torch.Tensor) -> torch.Tensor:
    """4개의 segment를 다시 21개 점 미래로 합친다.

    Args:
        segments: shape [N, 4, 6, 4].
            이웃 segment는 경계점 1개를 공유한다고 가정한다.

    Returns:
        shape [N, 21, 4].
            겹치는 경계점은 단순 평균으로 합친다.
    """
    if segments.ndim != 4 or segments.size(1) != 4 or segments.size(2) != 6 or segments.size(3) != 4:
        raise ValueError(f"segments must be [N, 4, 6, 4], got {tuple(segments.shape)}")

    n = segments.size(0)
    out = torch.zeros(n, 21, 4, device=segments.device, dtype=segments.dtype)
    cnt = torch.zeros(n, 21, 1, device=segments.device, dtype=segments.dtype)

    ranges = [(0, 6), (5, 11), (10, 16), (15, 21)]
    for seg_idx, (s, e) in enumerate(ranges):
        out[:, s:e] += segments[:, seg_idx]
        cnt[:, s:e] += 1.0

    return out / cnt.clamp_min(1.0)


def boundary_consistency_loss(segments: torch.Tensor) -> torch.Tensor:
    """인접 segment 경계점 일치 손실을 계산한다.

    Args:
        segments: shape [N, 4, 6, 4].

    Returns:
        scalar tensor.
    """
    if segments.numel() == 0:
        return segments.new_zeros(())
    loss = 0.0
    loss = loss + torch.mean((segments[:, 0, -1] - segments[:, 1, 0]) ** 2)
    loss = loss + torch.mean((segments[:, 1, -1] - segments[:, 2, 0]) ** 2)
    loss = loss + torch.mean((segments[:, 2, -1] - segments[:, 3, 0]) ** 2)
    return loss / 3.0


def build_linear_flow_path(clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """선형 flow path 중간 샘플을 만든다.

    Args:
        clean: shape [N, 4, 6, 4].
        noise: shape [N, 4, 6, 4].
        t: shape [N] 또는 [1]. 값 범위는 [0, 1].

    Returns:
        shape [N, 4, 6, 4].
    """
    if t.ndim == 1:
        t = t[:, None, None, None]
    return (1.0 - t) * noise + t * clean


@torch.no_grad()
def midpoint_ode_solve(
    z0: torch.Tensor,
    vector_field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    steps: int = 4,
) -> torch.Tensor:
    """2차 midpoint ODE 적분으로 clean sample을 만든다.

    Args:
        z0: shape [N, 4, 6, 4].
            시작 noise sample이다.
        vector_field_fn: (z, t) -> v 를 계산하는 함수.
            z shape는 [N, 4, 6, 4], t shape는 [N].
        steps: ODE 적분 step 수.

    Returns:
        shape [N, 4, 6, 4].
    """
    z = z0
    dt = 1.0 / float(steps)
    n = z.size(0)
    for i in range(steps):
        t0 = z.new_full((n,), float(i) * dt)
        k1 = vector_field_fn(z, t0)
        z_mid = z + 0.5 * dt * k1
        t_mid = z.new_full((n,), (float(i) + 0.5) * dt)
        k2 = vector_field_fn(z_mid, t_mid)
        z = z + dt * k2
    return z
