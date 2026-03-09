from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import torch


_POINT_TYPES: Tuple[str, ...] = (
    "DASH_SOLID_YELLOW",
    "DASH_SOLID_WHITE",
    "DASHED_WHITE",
    "DASHED_YELLOW",
    "DOUBLE_SOLID_YELLOW",
    "DOUBLE_SOLID_WHITE",
    "DOUBLE_DASH_YELLOW",
    "DOUBLE_DASH_WHITE",
    "SOLID_YELLOW",
    "SOLID_WHITE",
    "SOLID_DASH_WHITE",
    "SOLID_DASH_YELLOW",
    "EDGE",
    "NONE",
    "UNKNOWN",
    "CROSSWALK",
    "CENTERLINE",
)

_AGENT_TYPE_TO_NAME: Dict[int, str] = {
    0: "vehicle",
    1: "pedestrian",
    2: "cyclist",
}

_AGENT_TYPE_TO_COLOR: Dict[int, str] = {
    0: "#2E6BE6",
    1: "#FF8C42",
    2: "#2FA84F",
}

_MAP_STYLE: Dict[str, Dict[str, Any]] = {
    "lane": {
        "color": "#6C7A89",
        "linewidth": 1.0,
        "linestyle": (0, (5, 5)),
        "zorder": 2,
    },
    "road_edge": {
        "color": "#3A3A3A",
        "linewidth": 1.8,
        "linestyle": "-",
        "zorder": 3,
    },
    "crosswalk_edge": {
        "edgecolor": "#F5F5F5",
        "facecolor": (1.0, 1.0, 1.0, 0.08),
        "linewidth": 1.0,
        "zorder": 1,
        "hatch": "///",
    },
    "unknown_line": {
        "color": "#B0B0B0",
        "linewidth": 0.9,
        "linestyle": (0, (1, 3)),
        "zorder": 2,
    },
}


@dataclass
class RolloutVisualizationConfig:
    """평가 롤아웃 시각화 옵션."""

    enabled: bool = False
    output_dir: str = "rollout_vis"
    scenario_index_in_batch: int = 0
    max_scenarios: int = 1
    save_gap_sec: float = 0.1
    raw_step_sec: float = 0.1
    trajectory_mode: str = "line"  # point | line | box
    figsize: Tuple[float, float] = (12.0, 12.0)
    dpi: int = 180
    margin_m: float = 20.0
    video_fps: Optional[int] = None
    draw_agent_ids: bool = False

    def normalized_trajectory_mode(self) -> str:
        mode = str(self.trajectory_mode).strip().lower()
        if mode == "rectangle":
            return "box"
        if mode not in {"point", "line", "box"}:
            raise ValueError(f"Unsupported trajectory_mode: {self.trajectory_mode}")
        return mode


@dataclass
class RenderedScenarioArtifacts:
    """시나리오 1개를 렌더링한 결과물 경로."""

    scenario_id: str
    frame_dir: Path
    frame_paths: List[Path]
    mp4_path: Optional[Path]
    gif_path: Optional[Path]


def build_rollout_visualization_config(model_config: Any) -> RolloutVisualizationConfig:
    """모델 설정에서 시각화 옵션을 읽어 dataclass로 변환한다."""
    vis_cfg = getattr(model_config, "visualization", None)
    if vis_cfg is None:
        return RolloutVisualizationConfig(enabled=False)

    def _get(name: str, default: Any) -> Any:
        return getattr(vis_cfg, name, default)

    return RolloutVisualizationConfig(
        enabled=bool(_get("enabled", False)),
        output_dir=str(_get("output_dir", "rollout_vis")),
        scenario_index_in_batch=int(_get("scenario_index_in_batch", 0)),
        max_scenarios=int(_get("max_scenarios", 1)),
        save_gap_sec=float(_get("save_gap_sec", 0.1)),
        raw_step_sec=float(_get("raw_step_sec", 0.1)),
        trajectory_mode=str(_get("trajectory_mode", "line")),
        figsize=tuple(_get("figsize", (12.0, 12.0))),
        dpi=int(_get("dpi", 180)),
        margin_m=float(_get("margin_m", 20.0)),
        video_fps=_maybe_int(_get("video_fps", None)),
        draw_agent_ids=bool(_get("draw_agent_ids", False)),
    )


def render_rollout_visualization(
    scenario_data: Mapping[str, Any],
    rollout: Mapping[str, torch.Tensor],
    config: RolloutVisualizationConfig,
    batch_idx: int,
) -> RenderedScenarioArtifacts:
    """한 시나리오의 롤아웃 결과를 프레임들과 영상으로 저장한다."""
    scenario_id = _get_scenario_id(scenario_data, batch_idx=batch_idx)
    save_root = Path(config.output_dir).expanduser().resolve()
    frame_dir = save_root / scenario_id / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    observed = _extract_observed_agent_state(scenario_data)
    predicted = _extract_predicted_agent_state(rollout)
    ground_truth = _extract_ground_truth_agent_state(scenario_data, rollout)
    map_geoms = _extract_map_geometries(scenario_data)

    frame_indices = _select_frame_indices(
        num_future_steps=int(predicted["position"].shape[1]),
        save_gap_sec=float(config.save_gap_sec),
        raw_step_sec=float(config.raw_step_sec),
    )

    bounds = _compute_scene_bounds(
        map_geometries=map_geoms,
        observed=observed,
        predicted=predicted,
        ground_truth=ground_truth,
        margin_m=float(config.margin_m),
    )

    mode = config.normalized_trajectory_mode()
    frame_paths: List[Path] = []
    for save_frame_idx, raw_index in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        fig.patch.set_facecolor("#202020")
        _configure_axes(ax=ax, bounds=bounds)
        _draw_map(ax=ax, map_geometries=map_geoms)
        _draw_agents(
            ax=ax,
            scenario_data=scenario_data,
            observed=observed,
            predicted=predicted,
            ground_truth=ground_truth,
            current_future_index=raw_index,
            trajectory_mode=mode,
            draw_agent_ids=bool(config.draw_agent_ids),
        )
        elapsed_sec = float(raw_index + 1) * float(config.raw_step_sec)
        ax.set_title(f"scenario={scenario_id} | t={elapsed_sec:.2f}s | mode={mode}")
        _draw_legend(ax)

        frame_path = frame_dir / f"{save_frame_idx:05d}.png"
        fig.savefig(frame_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        frame_paths.append(frame_path)

    mp4_path: Optional[Path] = None
    gif_path: Optional[Path] = None
    if frame_paths:
        fps = int(config.video_fps) if config.video_fps is not None else max(1, int(round(1.0 / max(config.save_gap_sec, 1e-3))))
        mp4_path = save_root / scenario_id / f"{scenario_id}.mp4"
        gif_path = save_root / scenario_id / f"{scenario_id}.gif"
        _write_mp4_from_frames(frame_paths=frame_paths, output_path=mp4_path, fps=fps)
        _write_gif_from_frames(frame_paths=frame_paths, output_path=gif_path, fps=fps)

    return RenderedScenarioArtifacts(
        scenario_id=scenario_id,
        frame_dir=frame_dir,
        frame_paths=frame_paths,
        mp4_path=mp4_path,
        gif_path=gif_path,
    )


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return int(value)


def _get_scenario_id(scenario_data: Mapping[str, Any], batch_idx: int) -> str:
    try:
        scenario_id = scenario_data.get("scenario_id", None)
    except AttributeError:
        scenario_id = scenario_data["scenario_id"] if "scenario_id" in scenario_data else None
    if isinstance(scenario_id, (list, tuple)):
        scenario_id = scenario_id[0] if scenario_id else None
    if isinstance(scenario_id, torch.Tensor):
        if scenario_id.numel() == 1:
            scenario_id = str(scenario_id.item())
        else:
            scenario_id = None
    if scenario_id is None:
        return f"batch_{batch_idx:05d}"
    return str(scenario_id)


def _extract_observed_agent_state(scenario_data: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    agent = scenario_data["agent"]
    num_historical_steps = int(_safe_int(scenario_data["agent"]["valid_mask"].shape[1]))
    if "num_historical_steps" in scenario_data:
        num_historical_steps = int(_safe_int(scenario_data["num_historical_steps"]))
    # repo 설정상 history 길이는 model config에서 관리되므로, 현재 frame 끝 index는 valid_mask 길이에서 future rollout 길이를 제외해서 구한다.
    # 여기서는 downstream에서 GT 길이를 기준으로 다시 잘라서 쓸 수 있게 전체 observed prefix를 그대로 반환한다.
    return {
        "position": agent["position"],
        "heading": agent["heading"],
        "valid_mask": agent["valid_mask"],
        "shape": agent["shape"],
        "type": agent["type"],
        "id": agent.get("id", None),
        "av_index": agent.get("av_index", None),
        "num_historical_steps": num_historical_steps,
    }


def _extract_predicted_agent_state(rollout: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    pred_traj = rollout["pred_traj"]
    pred_head = rollout["pred_head"]
    pred_valid_mask = rollout.get("pred_valid_mask", None)
    if pred_valid_mask is None:
        pred_valid_mask = ~(pred_traj.abs().sum(dim=-1) == 0)
    return {
        "position": pred_traj,
        "heading": pred_head,
        "valid_mask": pred_valid_mask,
    }


def _extract_ground_truth_agent_state(
    scenario_data: Mapping[str, Any],
    rollout: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    agent = scenario_data["agent"]
    rollout_steps = int(rollout["pred_traj"].shape[1])
    num_total_steps = int(agent["position"].shape[1])
    num_historical_steps = num_total_steps - rollout_steps
    return {
        "position": agent["position"][:, num_historical_steps:num_historical_steps + rollout_steps, :2],
        "heading": agent["heading"][:, num_historical_steps:num_historical_steps + rollout_steps],
        "valid_mask": agent["valid_mask"][:, num_historical_steps:num_historical_steps + rollout_steps],
        "num_historical_steps": num_historical_steps,
    }


def _extract_map_geometries(scenario_data: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    if "map_point" not in scenario_data or "map_polygon" not in scenario_data:
        return {"lane": [], "road_line": [], "road_edge": [], "crosswalk": []}

    map_point = scenario_data["map_point"]
    edge_index = scenario_data[("map_point", "to", "map_polygon")]["edge_index"]
    point_pos = _to_numpy(map_point["position"])[:, :2]
    point_type = _to_numpy(map_point["type"]).astype(np.int64)
    polygon_index = _to_numpy(edge_index[1]).astype(np.int64)
    point_index = _to_numpy(edge_index[0]).astype(np.int64)

    grouped: MutableMapping[int, List[int]] = defaultdict(list)
    for pt_idx, poly_idx in zip(point_index.tolist(), polygon_index.tolist()):
        grouped[int(poly_idx)].append(int(pt_idx))

    geoms: Dict[str, List[Dict[str, Any]]] = {"lane": [], "road_line": [], "road_edge": [], "crosswalk": []}
    for poly_idx in sorted(grouped.keys()):
        indices = grouped[poly_idx]
        if not indices:
            continue
        polyline = point_pos[indices]
        if polyline.shape[0] < 2:
            continue
        type_idx = int(point_type[indices[0]])
        type_name = _POINT_TYPES[type_idx] if 0 <= type_idx < len(_POINT_TYPES) else "UNKNOWN"
        if type_name == "CENTERLINE":
            geoms["lane"].append({"xy": polyline, "type_name": type_name})
        elif type_name == "EDGE":
            geoms["road_edge"].append({"xy": polyline, "type_name": type_name})
        elif type_name == "CROSSWALK":
            geoms["crosswalk"].append({"xy": polyline, "type_name": type_name})
        elif type_name != "NONE":
            geoms["road_line"].append({"xy": polyline, "type_name": type_name})
    return geoms


def _select_frame_indices(num_future_steps: int, save_gap_sec: float, raw_step_sec: float) -> List[int]:
    if num_future_steps <= 0:
        return []
    if raw_step_sec <= 0.0:
        raise ValueError("raw_step_sec must be positive")
    if save_gap_sec <= 0.0:
        save_gap_sec = raw_step_sec

    indices: List[int] = []
    total_time = float(num_future_steps) * float(raw_step_sec)
    current_time = float(save_gap_sec)
    while current_time <= total_time + 1e-8:
        idx = max(0, int(math.ceil(current_time / raw_step_sec)) - 1)
        idx = min(num_future_steps - 1, idx)
        if not indices or idx != indices[-1]:
            indices.append(idx)
        current_time += float(save_gap_sec)

    if not indices:
        return [num_future_steps - 1]
    if indices[-1] != num_future_steps - 1:
        indices.append(num_future_steps - 1)
    return indices


def _compute_scene_bounds(
    map_geometries: Mapping[str, List[Dict[str, Any]]],
    observed: Mapping[str, torch.Tensor],
    predicted: Mapping[str, torch.Tensor],
    ground_truth: Mapping[str, torch.Tensor],
    margin_m: float,
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []

    for group_name in ("lane", "road_line", "road_edge", "crosswalk"):
        for geom in map_geometries.get(group_name, []):
            if geom["xy"].size == 0:
                continue
            xs.extend(geom["xy"][:, 0].tolist())
            ys.extend(geom["xy"][:, 1].tolist())

    obs_pos = _to_numpy(observed["position"])[..., :2]
    obs_valid = _to_numpy(observed["valid_mask"]).astype(bool)
    if obs_pos.size > 0 and obs_valid.any():
        xs.extend(obs_pos[..., 0][obs_valid].tolist())
        ys.extend(obs_pos[..., 1][obs_valid].tolist())

    pred_pos = _to_numpy(predicted["position"])
    pred_valid = _to_numpy(predicted["valid_mask"]).astype(bool)
    if pred_pos.size > 0 and pred_valid.any():
        xs.extend(pred_pos[..., 0][pred_valid].tolist())
        ys.extend(pred_pos[..., 1][pred_valid].tolist())

    gt_pos = _to_numpy(ground_truth["position"])
    gt_valid = _to_numpy(ground_truth["valid_mask"]).astype(bool)
    if gt_pos.size > 0 and gt_valid.any():
        xs.extend(gt_pos[..., 0][gt_valid].tolist())
        ys.extend(gt_pos[..., 1][gt_valid].tolist())

    if not xs or not ys:
        return (-50.0, 50.0, -50.0, 50.0)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return (xmin - margin_m, xmax + margin_m, ymin - margin_m, ymax + margin_m)


def _configure_axes(ax: plt.Axes, bounds: Tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = bounds
    ax.set_facecolor("#202020")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.axis("off")


def _draw_map(ax: plt.Axes, map_geometries: Mapping[str, List[Dict[str, Any]]]) -> None:
    for geom in map_geometries.get("crosswalk", []):
        xy = geom["xy"]
        if xy.shape[0] < 3:
            continue
        polygon_xy = np.concatenate([xy, xy[:1]], axis=0)
        patch = Polygon(
            polygon_xy,
            closed=True,
            fill=True,
            facecolor=_MAP_STYLE["crosswalk_edge"]["facecolor"],
            edgecolor=_MAP_STYLE["crosswalk_edge"]["edgecolor"],
            linewidth=_MAP_STYLE["crosswalk_edge"]["linewidth"],
            hatch=_MAP_STYLE["crosswalk_edge"]["hatch"],
            zorder=_MAP_STYLE["crosswalk_edge"]["zorder"],
        )
        ax.add_patch(patch)

    for geom in map_geometries.get("lane", []):
        xy = geom["xy"]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=_MAP_STYLE["lane"]["color"],
            linewidth=_MAP_STYLE["lane"]["linewidth"],
            linestyle=_MAP_STYLE["lane"]["linestyle"],
            zorder=_MAP_STYLE["lane"]["zorder"],
        )

    for geom in map_geometries.get("road_edge", []):
        xy = geom["xy"]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=_MAP_STYLE["road_edge"]["color"],
            linewidth=_MAP_STYLE["road_edge"]["linewidth"],
            linestyle=_MAP_STYLE["road_edge"]["linestyle"],
            zorder=_MAP_STYLE["road_edge"]["zorder"],
        )

    for geom in map_geometries.get("road_line", []):
        _draw_road_line(ax=ax, xy=geom["xy"], type_name=str(geom["type_name"]))


def _draw_road_line(ax: plt.Axes, xy: np.ndarray, type_name: str) -> None:
    color, primary_style, secondary_style, is_double = _road_line_style(type_name)
    if xy.shape[0] < 2:
        return
    if not is_double:
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.2, linestyle=primary_style, zorder=4)
        return

    normal_offset = 0.18
    first_segments, second_segments = _offset_polyline_pair(xy=xy, offset_m=normal_offset)
    for seg in first_segments:
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.0, linestyle=primary_style, zorder=4)
    for seg in second_segments:
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.0, linestyle=secondary_style, zorder=4)


def _road_line_style(type_name: str) -> Tuple[str, Any, Any, bool]:
    name = type_name.upper()
    if name in {"UNKNOWN", "NONE"}:
        style = _MAP_STYLE["unknown_line"]
        return style["color"], style["linestyle"], style["linestyle"], False

    color = "#FFFFFF"
    if "YELLOW" in name:
        color = "#F6C945"

    dashed = (0, (4, 4))
    solid = "-"

    if name in {"DASHED_WHITE", "DASHED_YELLOW"}:
        return color, dashed, dashed, False
    if name in {"SOLID_WHITE", "SOLID_YELLOW"}:
        return color, solid, solid, False
    if name in {"DOUBLE_SOLID_WHITE", "DOUBLE_SOLID_YELLOW", "DOUBLE_DASH_WHITE", "DOUBLE_DASH_YELLOW"}:
        style = dashed if "DASH" in name else solid
        return color, style, style, True
    if name in {"SOLID_DASH_WHITE", "SOLID_DASH_YELLOW"}:
        return color, solid, dashed, True
    if name in {"DASH_SOLID_WHITE", "DASH_SOLID_YELLOW"}:
        return color, dashed, solid, True
    return color, solid, solid, False


def _offset_polyline_pair(xy: np.ndarray, offset_m: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    first_segments: List[np.ndarray] = []
    second_segments: List[np.ndarray] = []
    for start, end in zip(xy[:-1], xy[1:]):
        delta = end - start
        length = float(np.hypot(delta[0], delta[1]))
        if length < 1e-6:
            continue
        normal = np.array([-delta[1], delta[0]], dtype=np.float64) / length
        offset = normal * float(offset_m)
        first_segments.append(np.stack([start + offset, end + offset], axis=0))
        second_segments.append(np.stack([start - offset, end - offset], axis=0))
    return first_segments, second_segments


def _draw_agents(
    ax: plt.Axes,
    scenario_data: Mapping[str, Any],
    observed: Mapping[str, torch.Tensor],
    predicted: Mapping[str, torch.Tensor],
    ground_truth: Mapping[str, torch.Tensor],
    current_future_index: int,
    trajectory_mode: str,
    draw_agent_ids: bool,
) -> None:
    agent = scenario_data["agent"]
    obs_pos = _to_numpy(observed["position"])[..., :2]
    obs_head = _to_numpy(observed["heading"])
    obs_valid = _to_numpy(observed["valid_mask"]).astype(bool)
    pred_pos = _to_numpy(predicted["position"])
    pred_head = _to_numpy(predicted["heading"])
    pred_valid = _to_numpy(predicted["valid_mask"]).astype(bool)
    gt_pos = _to_numpy(ground_truth["position"])
    gt_head = _to_numpy(ground_truth["heading"])
    gt_valid = _to_numpy(ground_truth["valid_mask"]).astype(bool)
    agent_type = _to_numpy(agent["type"]).astype(np.int64)
    agent_shape = _to_numpy(agent["shape"])
    agent_ids = agent.get("id", None)
    av_index = agent.get("av_index", None)
    av_index_int = int(av_index) if av_index is not None and not isinstance(av_index, torch.Tensor) else int(av_index.item()) if isinstance(av_index, torch.Tensor) else None

    rollout_steps = pred_pos.shape[1]
    num_total_steps = obs_pos.shape[1]
    num_historical_steps = num_total_steps - rollout_steps
    current_global_index = min(num_total_steps - 1, num_historical_steps + current_future_index)

    for agent_idx in range(obs_pos.shape[0]):
        agent_kind = int(agent_type[agent_idx])
        if agent_kind not in _AGENT_TYPE_TO_NAME:
            continue
        color = _AGENT_TYPE_TO_COLOR[agent_kind]

        past_mask = obs_valid[agent_idx, :num_historical_steps]
        past_xy = obs_pos[agent_idx, :num_historical_steps][past_mask]
        past_heading = obs_head[agent_idx, :num_historical_steps][past_mask]

        pred_mask = pred_valid[agent_idx, : current_future_index + 1]
        pred_xy = pred_pos[agent_idx, : current_future_index + 1][pred_mask]
        pred_heading = pred_head[agent_idx, : current_future_index + 1][pred_mask]

        gt_mask = gt_valid[agent_idx]
        gt_xy = gt_pos[agent_idx][gt_mask]
        gt_heading_valid = gt_head[agent_idx][gt_mask]

        _draw_traj(
            ax=ax,
            xy=past_xy,
            heading=past_heading,
            shape_xy=_agent_length_width(agent_shape, agent_idx, current_global_index),
            mode=trajectory_mode,
            color=color,
            alpha=0.35,
            linewidth=1.2,
            linestyle=(0, (2, 2)),
            zorder=10,
        )
        _draw_traj(
            ax=ax,
            xy=gt_xy,
            heading=gt_heading_valid,
            shape_xy=_agent_length_width(agent_shape, agent_idx, current_global_index),
            mode=trajectory_mode,
            color=color,
            alpha=0.55,
            linewidth=1.2,
            linestyle=(0, (5, 3)),
            zorder=11,
        )
        _draw_traj(
            ax=ax,
            xy=pred_xy,
            heading=pred_heading,
            shape_xy=_agent_length_width(agent_shape, agent_idx, current_global_index),
            mode=trajectory_mode,
            color=color,
            alpha=0.95,
            linewidth=1.8,
            linestyle="-",
            zorder=12,
        )

        current_xy, current_heading = _current_agent_pose(
            obs_pos=obs_pos[agent_idx],
            obs_head=obs_head[agent_idx],
            obs_valid=obs_valid[agent_idx],
            pred_pos=pred_pos[agent_idx],
            pred_head=pred_head[agent_idx],
            pred_valid=pred_valid[agent_idx],
            gt_pos=gt_pos[agent_idx],
            gt_head=gt_head[agent_idx],
            gt_valid=gt_valid[agent_idx],
            current_future_index=current_future_index,
            num_historical_steps=num_historical_steps,
        )
        if current_xy is not None and current_heading is not None:
            is_av = av_index_int is not None and int(agent_idx) == av_index_int
            edge_color = "#000000" if not is_av else "#FF3B30"
            _draw_current_agent_body(
                ax=ax,
                center_xy=current_xy,
                heading=float(current_heading),
                shape_xy=_agent_length_width(agent_shape, agent_idx, current_global_index),
                fill_color=color,
                edge_color=edge_color,
                zorder=20 if is_av else 18,
            )
            if draw_agent_ids and agent_ids is not None:
                label = str(agent_ids[agent_idx])
                ax.text(
                    current_xy[0],
                    current_xy[1] + 1.2,
                    label,
                    color="#FFFFFF",
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    zorder=25,
                )


def _draw_traj(
    ax: plt.Axes,
    xy: np.ndarray,
    heading: np.ndarray,
    shape_xy: Tuple[float, float],
    mode: str,
    color: str,
    alpha: float,
    linewidth: float,
    linestyle: Any,
    zorder: int,
) -> None:
    if xy.size == 0:
        return
    if mode == "point":
        ax.scatter(xy[:, 0], xy[:, 1], s=10.0, color=color, alpha=alpha, zorder=zorder)
        return
    if mode == "line":
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
        return
    if mode == "box":
        length, width = shape_xy
        for point_xy, yaw in zip(xy, heading):
            box = _oriented_box_corners(point_xy, float(yaw), length=length, width=width)
            patch = Polygon(
                box,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=max(0.7, linewidth),
                alpha=alpha,
                zorder=zorder,
            )
            ax.add_patch(patch)
            head_end = point_xy + 0.5 * length * np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float64)
            ax.plot(
                [point_xy[0], head_end[0]],
                [point_xy[1], head_end[1]],
                color=color,
                alpha=alpha,
                linewidth=max(0.7, linewidth),
                linestyle="-",
                zorder=zorder,
            )
        return
    raise ValueError(f"Unsupported trajectory mode: {mode}")


def _draw_current_agent_body(
    ax: plt.Axes,
    center_xy: np.ndarray,
    heading: float,
    shape_xy: Tuple[float, float],
    fill_color: str,
    edge_color: str,
    zorder: int,
) -> None:
    length, width = shape_xy
    box = _oriented_box_corners(center_xy, heading, length=length, width=width)
    patch = Polygon(
        box,
        closed=True,
        fill=True,
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=1.0,
        alpha=0.85,
        zorder=zorder,
    )
    ax.add_patch(patch)
    head_end = center_xy + 0.6 * length * np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    ax.plot(
        [center_xy[0], head_end[0]],
        [center_xy[1], head_end[1]],
        color=edge_color,
        linewidth=1.1,
        linestyle="-",
        zorder=zorder + 1,
    )


def _current_agent_pose(
    obs_pos: np.ndarray,
    obs_head: np.ndarray,
    obs_valid: np.ndarray,
    pred_pos: np.ndarray,
    pred_head: np.ndarray,
    pred_valid: np.ndarray,
    gt_pos: np.ndarray,
    gt_head: np.ndarray,
    gt_valid: np.ndarray,
    current_future_index: int,
    num_historical_steps: int,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if current_future_index >= 0 and current_future_index < pred_valid.shape[0] and pred_valid[current_future_index]:
        return pred_pos[current_future_index], float(pred_head[current_future_index])

    current_global_index = min(obs_pos.shape[0] - 1, num_historical_steps + current_future_index)
    if current_global_index >= 0 and current_global_index < obs_valid.shape[0] and obs_valid[current_global_index]:
        return obs_pos[current_global_index, :2], float(obs_head[current_global_index])

    if current_future_index >= 0 and current_future_index < gt_valid.shape[0] and gt_valid[current_future_index]:
        return gt_pos[current_future_index], float(gt_head[current_future_index])

    valid_indices = np.where(obs_valid[:num_historical_steps])[0]
    if valid_indices.size == 0:
        return None, None
    last_idx = int(valid_indices[-1])
    return obs_pos[last_idx, :2], float(obs_head[last_idx])


def _agent_length_width(agent_shape: np.ndarray, agent_idx: int, time_idx: int) -> Tuple[float, float]:
    if agent_shape.ndim == 3:
        safe_time_idx = min(max(int(time_idx), 0), agent_shape.shape[1] - 1)
        length = float(agent_shape[agent_idx, safe_time_idx, 0])
        width = float(agent_shape[agent_idx, safe_time_idx, 1])
    else:
        length = float(agent_shape[agent_idx, 0])
        width = float(agent_shape[agent_idx, 1])
    length = max(length, 0.8)
    width = max(width, 0.4)
    return length, width


def _oriented_box_corners(center_xy: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    half_length = 0.5 * float(length)
    half_width = 0.5 * float(width)
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cos_yaw = math.cos(heading)
    sin_yaw = math.sin(heading)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return (local @ rotation.T) + center_xy.reshape(1, 2)


def _draw_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], color=_MAP_STYLE["lane"]["color"], linestyle=_MAP_STYLE["lane"]["linestyle"], linewidth=1.0, label="lane"),
        Line2D([0], [0], color="#FFFFFF", linestyle="-", linewidth=1.0, label="road_line"),
        Line2D([0], [0], color=_MAP_STYLE["road_edge"]["color"], linestyle="-", linewidth=1.5, label="road_edge"),
        Line2D([0], [0], color="#F5F5F5", linestyle="-", linewidth=1.0, label="crosswalk"),
        Line2D([0], [0], color=_AGENT_TYPE_TO_COLOR[0], linestyle="-", linewidth=1.6, label="vehicle"),
        Line2D([0], [0], color=_AGENT_TYPE_TO_COLOR[1], linestyle="-", linewidth=1.6, label="pedestrian"),
        Line2D([0], [0], color=_AGENT_TYPE_TO_COLOR[2], linestyle="-", linewidth=1.6, label="cyclist"),
    ]
    legend = ax.legend(handles=handles, loc="upper right", framealpha=0.85, fontsize=7)
    legend.get_frame().set_facecolor("#101010")
    legend.get_frame().set_edgecolor("#303030")
    for text in legend.get_texts():
        text.set_color("#F5F5F5")


def _write_mp4_from_frames(frame_paths: Sequence[Path], output_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("MP4 저장에는 imageio와 imageio-ffmpeg가 필요합니다.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=int(fps), codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            frame = _pad_frame_to_even_size(frame)
            writer.append_data(frame)


def _write_gif_from_frames(frame_paths: Sequence[Path], output_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("GIF 저장에는 imageio가 필요합니다.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / float(max(1, int(fps)))
    with imageio.get_writer(str(output_path), mode="I", duration=duration) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


def _pad_frame_to_even_size(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)


def _safe_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
