from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from .benchmarks import CASE33_EDGE_DATA, RadialBenchmark, case33, case69


@dataclass(frozen=True)
class Topology:
    topology_id: int
    edge_indices: tuple[int, ...]
    label: str


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def enumerate_case33_topologies() -> list[Topology]:
    """Enumerate unique radial topologies reachable by one tie closure and one cycle opening."""
    data = CASE33_EDGE_DATA.copy()
    all_edges = data[:, :2].astype(int) - 1
    base_idx = tuple(np.where(data[:, 4] == 1)[0].tolist())
    tie_idx = np.where(data[:, 4] == 0)[0]
    n_bus = 33
    seen: dict[frozenset[int], str] = {frozenset(base_idx): "base"}

    base_graph = nx.Graph()
    base_graph.add_nodes_from(range(n_bus))
    base_graph.add_edges_from([tuple(all_edges[i]) for i in base_idx])
    assert nx.is_tree(base_graph)

    for tie in tie_idx:
        u, v = all_edges[tie]
        path = nx.shortest_path(base_graph, source=int(u), target=int(v))
        cycle_edges = {_edge_key(path[i], path[i + 1]) for i in range(len(path) - 1)}
        edge_to_idx = {_edge_key(*all_edges[i]): i for i in base_idx}
        for cycle_edge in cycle_edges:
            opened = edge_to_idx[cycle_edge]
            candidate = set(base_idx)
            candidate.remove(opened)
            candidate.add(int(tie))
            graph = nx.Graph()
            graph.add_nodes_from(range(n_bus))
            graph.add_edges_from([tuple(all_edges[i]) for i in candidate])
            if nx.is_tree(graph):
                seen.setdefault(
                    frozenset(candidate),
                    f"close_{all_edges[tie,0]+1}-{all_edges[tie,1]+1}_open_{all_edges[opened,0]+1}-{all_edges[opened,1]+1}",
                )

    ordered = sorted(seen.items(), key=lambda item: (item[1] != "base", item[1]))
    return [Topology(i, tuple(sorted(edges)), label) for i, (edges, label) in enumerate(ordered)]


def topology_edges(benchmark: RadialBenchmark, topology: Topology | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if benchmark.name == "MATPOWER case33bw":
        if topology is None:
            topology = enumerate_case33_topologies()[0]
        edge_data = CASE33_EDGE_DATA[np.array(topology.edge_indices)]
        edges = edge_data[:, :2].astype(int) - 1
        resistance = edge_data[:, 2].astype(float)
        reactance = edge_data[:, 3].astype(float)
        return edges, resistance, reactance
    return benchmark.closed_edges.copy(), benchmark.resistance_ohm.copy(), benchmark.reactance_ohm.copy()


def orient_tree(n_bus: int, edges: np.ndarray, root: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_bus))
    graph.add_edges_from([tuple(map(int, e)) for e in edges])
    if not nx.is_tree(graph):
        raise ValueError("Power-flow solver requires a connected radial topology.")
    parent = np.full(n_bus, -1, dtype=int)
    parent_edge = np.full(n_bus, -1, dtype=int)
    depth = np.zeros(n_bus, dtype=int)
    edge_map = {_edge_key(int(u), int(v)): i for i, (u, v) in enumerate(edges)}
    order = [root]
    for p, c in nx.bfs_edges(graph, root):
        parent[c] = p
        parent_edge[c] = edge_map[_edge_key(p, c)]
        depth[c] = depth[p] + 1
        order.append(c)
    return parent, parent_edge, np.asarray(order, dtype=int)


def radial_power_flow(
    benchmark: RadialBenchmark,
    p_kw: np.ndarray,
    q_kvar: np.ndarray,
    topology: Topology | None = None,
    slack_vm: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> dict[str, np.ndarray | float | bool | int]:
    n_bus = len(benchmark.bus_p_kw)
    edges, r_ohm, x_ohm = topology_edges(benchmark, topology)
    parent, parent_edge, order = orient_tree(n_bus, edges)
    z_base = (benchmark.base_kv * 1e3) ** 2 / (benchmark.base_mva * 1e6)
    z_pu = (r_ohm + 1j * x_ohm) / z_base
    s_pu = (np.asarray(p_kw) + 1j * np.asarray(q_kvar)) / (benchmark.base_mva * 1000.0)
    v = np.ones(n_bus, dtype=complex) * complex(slack_vm, 0.0)
    branch_current = np.zeros(len(edges), dtype=complex)

    converged = False
    for iteration in range(1, max_iter + 1):
        safe_v = np.where(np.abs(v) < 1e-8, 1 + 0j, v)
        i_bus = np.conj(s_pu / safe_v)
        downstream = i_bus.copy()
        branch_current.fill(0.0)
        for node in order[:0:-1]:
            edge_index = parent_edge[node]
            branch_current[edge_index] = downstream[node]
            downstream[parent[node]] += downstream[node]
        v_new = np.empty_like(v)
        v_new[0] = complex(slack_vm, 0.0)
        for node in order[1:]:
            edge_index = parent_edge[node]
            v_new[node] = v_new[parent[node]] - z_pu[edge_index] * branch_current[edge_index]
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            converged = True
            break
        v = v_new

    losses_kw = float(np.sum((r_ohm / z_base) * np.abs(branch_current) ** 2) * benchmark.base_mva * 1000.0)
    vm = np.abs(v)
    return {
        "vm": vm,
        "vmin": float(vm.min()),
        "vmax": float(vm.max()),
        "loss_kw": losses_kw,
        "converged": converged,
        "iterations": iteration,
        "branch_current_pu": np.abs(branch_current),
    }


def linear_distflow_proxy(
    benchmark: RadialBenchmark,
    p_kw: np.ndarray,
    q_kvar: np.ndarray,
    topology: Topology | None = None,
    slack_vm: float = 1.0,
) -> dict[str, float | np.ndarray]:
    n_bus = len(benchmark.bus_p_kw)
    edges, r_ohm, x_ohm = topology_edges(benchmark, topology)
    parent, parent_edge, order = orient_tree(n_bus, edges)
    z_base = (benchmark.base_kv * 1e3) ** 2 / (benchmark.base_mva * 1e6)
    r_pu = r_ohm / z_base
    x_pu = x_ohm / z_base
    p_pu = np.asarray(p_kw, dtype=float) / (benchmark.base_mva * 1000.0)
    q_pu = np.asarray(q_kvar, dtype=float) / (benchmark.base_mva * 1000.0)
    p_down = p_pu.copy()
    q_down = q_pu.copy()
    p_flow = np.zeros(len(edges))
    q_flow = np.zeros(len(edges))
    for node in order[:0:-1]:
        edge_index = parent_edge[node]
        p_flow[edge_index] = p_down[node]
        q_flow[edge_index] = q_down[node]
        p_down[parent[node]] += p_down[node]
        q_down[parent[node]] += q_down[node]
    v_sq = np.ones(n_bus) * slack_vm**2
    for node in order[1:]:
        edge_index = parent_edge[node]
        v_sq[node] = max(
            1e-6,
            v_sq[parent[node]] - 2.0 * (r_pu[edge_index] * p_flow[edge_index] + x_pu[edge_index] * q_flow[edge_index]),
        )
    vm = np.sqrt(v_sq)
    # Branch flows use edge order; BFS node order may differ after reconfiguration.
    sending_bus = np.empty(len(edges), dtype=int)
    for node in order[1:]:
        sending_bus[parent_edge[node]] = parent[node]
    loss_pu = np.sum(
        r_pu * (p_flow**2 + q_flow**2)
        / np.maximum(v_sq[sending_bus], 1e-6)
    )
    return {"vm": vm, "vmin": float(vm.min()), "vmax": float(vm.max()), "loss_kw": float(loss_pu * benchmark.base_mva * 1000.0)}


def graph_matrices(benchmark: RadialBenchmark, topology: Topology | None, max_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_bus = len(benchmark.bus_p_kw)
    edges, r_ohm, x_ohm = topology_edges(benchmark, topology)
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    impedance = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    for (u, v), r, x in zip(edges, r_ohm, x_ohm):
        u, v = int(u), int(v)
        adjacency[u, v] = adjacency[v, u] = 1.0
        impedance[u, v] = impedance[v, u] = float(np.hypot(r, x))
    mask = np.zeros(max_nodes, dtype=np.float32)
    mask[:n_bus] = 1.0
    return adjacency, impedance, mask


def generate_load_state(
    benchmark: RadialBenchmark,
    rng: np.random.Generator,
    global_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    hour = int(rng.integers(0, 24))
    day = int(rng.integers(1, 366))
    weekday = int(rng.integers(0, 7))
    daily = 0.80 + 0.18 * np.sin(2 * np.pi * (hour - 8) / 24) + 0.08 * np.sin(4 * np.pi * (hour - 7) / 24)
    seasonal = 0.96 + 0.12 * np.cos(2 * np.pi * (day - 20) / 365)
    weekend = 0.93 if weekday >= 5 else 1.0
    if global_scale is None:
        global_scale = float(rng.uniform(0.72, 1.30))
    nodal = rng.lognormal(mean=-0.5 * 0.10**2, sigma=0.10, size=len(benchmark.bus_p_kw))
    scale = np.clip(daily * seasonal * weekend * global_scale * nodal, 0.35, 1.65)
    p = benchmark.bus_p_kw * scale
    q = benchmark.bus_q_kvar * scale * rng.normal(1.0, 0.025, size=len(scale))
    q[0] = 0.0
    return p, q, {"hour": hour, "day": day, "weekday": weekday, "global_scale": global_scale}


def corrupt_telemetry(
    p_kw: np.ndarray,
    q_kvar: np.ndarray,
    rng: np.random.Generator,
    max_missing: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bus = len(p_kw)
    missing_rate = float(rng.uniform(0.0, max_missing))
    observed = rng.random(n_bus) >= missing_rate
    observed[0] = True
    age = rng.exponential(scale=float(rng.uniform(0.1, 1.0)), size=n_bus)
    age = np.clip(age, 0.0, 4.0)
    noise_scale = 0.005 + 0.012 * age
    p_obs = p_kw * (1.0 + rng.normal(0.0, noise_scale))
    q_obs = q_kvar * (1.0 + rng.normal(0.0, noise_scale))
    p_obs[~observed] = np.nan
    q_obs[~observed] = np.nan
    return p_obs, q_obs, observed.astype(float), age


def impute_observations(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(observed, dtype=float).copy()
    missing = ~np.isfinite(values)
    if missing.any():
        finite_total = np.nansum(values)
        ref_total = np.sum(reference[~missing])
        scale = finite_total / ref_total if ref_total > 1e-9 else 1.0
        values[missing] = reference[missing] * scale
    return values


def topology_descriptor(benchmark: RadialBenchmark, topology: Topology | None) -> dict[str, float]:
    n_bus = len(benchmark.bus_p_kw)
    edges, r_ohm, x_ohm = topology_edges(benchmark, topology)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_bus))
    graph.add_edges_from([tuple(map(int, e)) for e in edges])
    depths = nx.single_source_shortest_path_length(graph, 0)
    degree = np.array([graph.degree(i) for i in range(n_bus)], dtype=float)
    return {
        "n_bus": float(n_bus),
        "depth_max": float(max(depths.values())),
        "depth_mean": float(np.mean(list(depths.values()))),
        "degree_mean": float(degree.mean()),
        "degree_max": float(degree.max()),
        "r_sum": float(np.sum(r_ohm)),
        "x_sum": float(np.sum(x_ohm)),
        "z_max": float(np.max(np.hypot(r_ohm, x_ohm))),
    }


def build_node_features(
    benchmark: RadialBenchmark,
    topology: Topology | None,
    p_obs: np.ndarray,
    q_obs: np.ndarray,
    observed_mask: np.ndarray,
    age: np.ndarray,
    max_nodes: int,
) -> np.ndarray:
    n_bus = len(benchmark.bus_p_kw)
    p_imp = impute_observations(p_obs, benchmark.bus_p_kw)
    q_imp = impute_observations(q_obs, benchmark.bus_q_kvar)
    edges, _, _ = topology_edges(benchmark, topology)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_bus))
    graph.add_edges_from([tuple(map(int, e)) for e in edges])
    degree = np.array([graph.degree(i) for i in range(n_bus)], dtype=float)
    depths = nx.single_source_shortest_path_length(graph, 0)
    depth = np.array([depths[i] for i in range(n_bus)], dtype=float)
    p_scale = max(float(np.max(benchmark.bus_p_kw)), 1.0)
    q_scale = max(float(np.max(np.abs(benchmark.bus_q_kvar))), 1.0)
    node_features = np.zeros((max_nodes, 10), dtype=np.float32)
    node_features[:n_bus, 0] = p_imp / p_scale
    node_features[:n_bus, 1] = q_imp / q_scale
    node_features[:n_bus, 2] = benchmark.bus_p_kw / p_scale
    node_features[:n_bus, 3] = benchmark.bus_q_kvar / q_scale
    node_features[:n_bus, 4] = degree / max(degree.max(), 1.0)
    node_features[:n_bus, 5] = depth / max(depth.max(), 1.0)
    node_features[:n_bus, 6] = observed_mask
    node_features[:n_bus, 7] = np.clip(age / 4.0, 0.0, 1.0)
    node_features[0, 8] = 1.0
    node_features[:n_bus, 9] = 1.0
    return node_features


def scenario_tabular_features(
    benchmark: RadialBenchmark,
    topology: Topology | None,
    p_obs: np.ndarray,
    q_obs: np.ndarray,
    observed_mask: np.ndarray,
    age: np.ndarray,
    slack_vm: float,
) -> tuple[np.ndarray, dict[str, float]]:
    p_imp = impute_observations(p_obs, benchmark.bus_p_kw)
    q_imp = impute_observations(q_obs, benchmark.bus_q_kvar)
    proxy = linear_distflow_proxy(benchmark, p_imp, q_imp, topology, slack_vm)
    desc = topology_descriptor(benchmark, topology)
    p_nonzero = p_imp[1:]
    q_nonzero = q_imp[1:]
    feature_dict = {
        "total_p_kw": float(p_imp.sum()),
        "total_q_kvar": float(q_imp.sum()),
        "mean_p_kw": float(p_nonzero.mean()),
        "std_p_kw": float(p_nonzero.std()),
        "max_p_kw": float(p_nonzero.max()),
        "mean_q_kvar": float(q_nonzero.mean()),
        "std_q_kvar": float(q_nonzero.std()),
        "slack_vm": float(slack_vm),
        "missing_rate": float(1.0 - observed_mask.mean()),
        "mean_age": float(age.mean()),
        "max_age": float(age.max()),
        "proxy_vmin": float(proxy["vmin"]),
        "proxy_loss_kw": float(proxy["loss_kw"]),
        **desc,
    }
    names = list(feature_dict.keys())
    return np.array([feature_dict[name] for name in names], dtype=np.float32), feature_dict


def make_dataset(
    output_dir: str | Path,
    n_case33: int = 5000,
    n_case69: int = 1200,
    seed: int = 20260922,
    max_nodes: int = 69,
) -> dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    benchmark33 = case33()
    benchmark69 = case69()
    topologies = enumerate_case33_topologies()
    topology_ids = np.arange(len(topologies))
    rng.shuffle(topology_ids)
    n_train_top = max(1, int(0.55 * len(topologies)))
    n_selection_top = max(1, int(0.15 * len(topologies)))
    n_calibration_top = max(1, int(0.15 * len(topologies)))
    train_top = set(topology_ids[:n_train_top].tolist())
    selection_top = set(topology_ids[n_train_top:n_train_top+n_selection_top].tolist())
    calibration_top = set(topology_ids[n_train_top+n_selection_top:n_train_top+n_selection_top+n_calibration_top].tolist())
    test_top = set(topology_ids[n_train_top+n_selection_top+n_calibration_top:].tolist())

    feature_names: list[str] | None = None
    records: list[dict[str, float | int | str]] = []
    node_arrays: list[np.ndarray] = []
    adj_arrays: list[np.ndarray] = []
    node_masks: list[np.ndarray] = []
    tab_arrays: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for sample_id in range(n_case33):
        topology = topologies[int(rng.integers(0, len(topologies)))]
        p_true, q_true, state = generate_load_state(benchmark33, rng)
        slack_vm = float(rng.uniform(0.975, 1.045))
        exact = radial_power_flow(benchmark33, p_true, q_true, topology, slack_vm)
        p_obs, q_obs, obs_mask, age = corrupt_telemetry(p_true, q_true, rng)
        tab, tab_dict = scenario_tabular_features(benchmark33, topology, p_obs, q_obs, obs_mask, age, slack_vm)
        if feature_names is None:
            feature_names = list(tab_dict.keys())
        node = build_node_features(benchmark33, topology, p_obs, q_obs, obs_mask, age, max_nodes)
        adj, _, nmask = graph_matrices(benchmark33, topology, max_nodes)
        if topology.topology_id in train_top:
            split = "train"
        elif topology.topology_id in selection_top:
            split = "selection"
        elif topology.topology_id in calibration_top:
            split = "calibration"
        else:
            split = "test"
        feasible = int(bool(exact["converged"]) and exact["vmin"] >= 0.95 and exact["vmax"] <= 1.05)
        records.append({
            "sample_id": sample_id,
            "network": "case33bw",
            "split": split,
            "topology_id": topology.topology_id,
            "hour": state["hour"],
            "day": state["day"],
            "weekday": state["weekday"],
            "global_scale": state["global_scale"],
            "slack_vm": slack_vm,
            "missing_rate": 1.0 - obs_mask.mean(),
            "mean_age": age.mean(),
            "target_vmin": exact["vmin"],
            "target_loss_kw": exact["loss_kw"],
            "target_feasible": feasible,
            "converged": int(exact["converged"]),
        })
        node_arrays.append(node)
        adj_arrays.append(adj)
        node_masks.append(nmask)
        tab_arrays.append(tab)
        targets.append(np.array([exact["vmin"], exact["loss_kw"], feasible], dtype=np.float32))

    offset = n_case33
    for j in range(n_case69):
        p_true, q_true, state = generate_load_state(benchmark69, rng)
        slack_vm = float(rng.uniform(0.98, 1.04))
        exact = radial_power_flow(benchmark69, p_true, q_true, None, slack_vm)
        p_obs, q_obs, obs_mask, age = corrupt_telemetry(p_true, q_true, rng)
        tab, tab_dict = scenario_tabular_features(benchmark69, None, p_obs, q_obs, obs_mask, age, slack_vm)
        node = build_node_features(benchmark69, None, p_obs, q_obs, obs_mask, age, max_nodes)
        adj, _, nmask = graph_matrices(benchmark69, None, max_nodes)
        feasible = int(bool(exact["converged"]) and exact["vmin"] >= 0.95 and exact["vmax"] <= 1.05)
        records.append({
            "sample_id": offset + j,
            "network": "case69",
            "split": "external",
            "topology_id": -1,
            "hour": state["hour"],
            "day": state["day"],
            "weekday": state["weekday"],
            "global_scale": state["global_scale"],
            "slack_vm": slack_vm,
            "missing_rate": 1.0 - obs_mask.mean(),
            "mean_age": age.mean(),
            "target_vmin": exact["vmin"],
            "target_loss_kw": exact["loss_kw"],
            "target_feasible": feasible,
            "converged": int(exact["converged"]),
        })
        node_arrays.append(node)
        adj_arrays.append(adj)
        node_masks.append(nmask)
        tab_arrays.append(tab)
        targets.append(np.array([exact["vmin"], exact["loss_kw"], feasible], dtype=np.float32))

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "scenario_metadata.csv", index=False)
    topology_frame = pd.DataFrame([
        {"topology_id": t.topology_id, "label": t.label, "n_edges": len(t.edge_indices), "split": ("train" if t.topology_id in train_top else "selection" if t.topology_id in selection_top else "calibration" if t.topology_id in calibration_top else "test")}
        for t in topologies
    ])
    topology_frame.to_csv(output_dir / "case33_topologies.csv", index=False)
    np.savez_compressed(
        output_dir / "scenario_arrays.npz",
        node_features=np.stack(node_arrays),
        adjacency=np.stack(adj_arrays),
        node_mask=np.stack(node_masks),
        tabular=np.stack(tab_arrays),
        targets=np.stack(targets),
        feature_names=np.array(feature_names, dtype=object),
    )
    return {"case33": n_case33, "case69": n_case69, "topologies": len(topologies)}
