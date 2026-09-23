"""Checkpoint loading, residual intervals and decision-candidate preparation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .benchmarks import case33
from .gridstudy import (
    Topology,
    build_node_features,
    corrupt_telemetry,
    graph_matrices,
    radial_power_flow,
    scenario_tabular_features,
)
from .models import DatasetBundle, GraphOutcomeModel, TrainedGraphModel


def load_graph_checkpoint(path: str | Path, bundle: DatasetBundle) -> TrainedGraphModel:
    payload = torch.load(path, weights_only=False)
    cfg = {
        k: v
        for k, v in payload["config"].items()
        if k not in {"proxy_v_index", "proxy_loss_index"}
    }
    model = GraphOutcomeModel(payload["node_dim"], payload["tab_dim"], **cfg)
    model.load_state_dict(payload["state_dict"])
    return TrainedGraphModel(
        model, payload["tab_mean"], payload["tab_std"], pd.DataFrame(), cfg
    )


def conformal_intervals(
    calibration_true: np.ndarray,
    calibration_pred: np.ndarray,
    test_pred: np.ndarray,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(calibration_true)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    v_resid = np.abs(calibration_true[:, 0] - calibration_pred[:, 0])
    q_v = float(np.quantile(v_resid, level, method="higher"))
    cal_log_true = np.log1p(np.maximum(calibration_true[:, 1], 0.0))
    cal_log_pred = np.log1p(np.maximum(calibration_pred[:, 1], 0.0))
    q_l = float(
        np.quantile(np.abs(cal_log_true - cal_log_pred), level, method="higher")
    )
    lower_v = test_pred[:, 0] - q_v
    upper_v = test_pred[:, 0] + q_v
    test_log = np.log1p(np.maximum(test_pred[:, 1], 0.0))
    lower_l = np.expm1(test_log - q_l).clip(min=0)
    upper_l = np.expm1(test_log + q_l)
    return lower_v, upper_v, lower_l, upper_l


def build_candidate_bundle(
    base_bundle: DatasetBundle,
    p_true: np.ndarray,
    q_true: np.ndarray,
    p_obs: np.ndarray,
    q_obs: np.ndarray,
    obs_mask: np.ndarray,
    age: np.ndarray,
    candidates: list[tuple[Topology, float]],
) -> tuple[DatasetBundle, pd.DataFrame]:
    benchmark = case33()
    max_nodes = base_bundle.node_features.shape[1]
    nodes, adjs, masks, tabs, targets, rows = [], [], [], [], [], []
    for j, (topology, slack_vm) in enumerate(candidates):
        exact = radial_power_flow(benchmark, p_true, q_true, topology, slack_vm)
        tab, _ = scenario_tabular_features(
            benchmark, topology, p_obs, q_obs, obs_mask, age, slack_vm
        )
        node = build_node_features(
            benchmark, topology, p_obs, q_obs, obs_mask, age, max_nodes
        )
        adj, _, nmask = graph_matrices(benchmark, topology, max_nodes)
        feasible = int(
            exact["converged"] and exact["vmin"] >= 0.95 and exact["vmax"] <= 1.05
        )
        nodes.append(node)
        adjs.append(adj)
        masks.append(nmask)
        tabs.append(tab)
        targets.append(
            np.array([exact["vmin"], exact["loss_kw"], feasible], dtype=np.float32)
        )
        rows.append(
            {
                "candidate": j,
                "topology_id": topology.topology_id,
                "slack_vm": slack_vm,
                "exact_vmin": exact["vmin"],
                "exact_loss_kw": exact["loss_kw"],
                "exact_feasible": feasible,
            }
        )
    metadata = pd.DataFrame(
        {"split": ["test"] * len(candidates), "network": ["case33bw"] * len(candidates)}
    )
    bundle = DatasetBundle(
        metadata,
        np.stack(nodes),
        np.stack(adjs),
        np.stack(masks),
        np.stack(tabs),
        np.stack(targets),
        base_bundle.feature_names,
    )
    return bundle, pd.DataFrame(rows)
