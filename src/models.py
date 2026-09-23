from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetBundle:
    metadata: pd.DataFrame
    node_features: np.ndarray
    adjacency: np.ndarray
    node_mask: np.ndarray
    tabular: np.ndarray
    targets: np.ndarray
    feature_names: list[str]


def load_bundle(processed_dir: str | Path) -> DatasetBundle:
    processed_dir = Path(processed_dir)
    metadata = pd.read_csv(processed_dir / "scenario_metadata.csv")
    arrays = np.load(processed_dir / "scenario_arrays.npz", allow_pickle=True)
    return DatasetBundle(
        metadata=metadata,
        node_features=arrays["node_features"].astype(np.float32),
        adjacency=arrays["adjacency"].astype(np.float32),
        node_mask=arrays["node_mask"].astype(np.float32),
        tabular=arrays["tabular"].astype(np.float32),
        targets=arrays["targets"].astype(np.float32),
        feature_names=[str(x) for x in arrays["feature_names"].tolist()],
    )


def split_indices(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    return {
        split: np.where(bundle.metadata["split"].to_numpy() == split)[0]
        for split in ["train", "selection", "calibration", "test", "external"]
    }


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, feasible_prob: np.ndarray | None = None) -> dict[str, float]:
    result = {
        "vmin_rmse": float(mean_squared_error(y_true[:, 0], y_pred[:, 0]) ** 0.5),
        "vmin_mae": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "vmin_r2": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "loss_rmse_kw": float(mean_squared_error(y_true[:, 1], y_pred[:, 1]) ** 0.5),
        "loss_mae_kw": float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        "loss_r2": float(r2_score(y_true[:, 1], y_pred[:, 1])),
    }
    if feasible_prob is not None and len(np.unique(y_true[:, 2])) > 1:
        result["feasible_auroc"] = float(roc_auc_score(y_true[:, 2], feasible_prob))
        result["feasible_f1"] = float(f1_score(y_true[:, 2], feasible_prob >= 0.5))
    else:
        result["feasible_auroc"] = np.nan
        result["feasible_f1"] = np.nan
    return result


def fit_tabular_baselines(bundle: DatasetBundle, output_dir: str | Path, seed: int = 20260922) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    idx = split_indices(bundle)
    x = bundle.tabular
    y_reg = np.column_stack([bundle.targets[:, 0], np.log1p(np.maximum(bundle.targets[:, 1], 0.0))])
    y_cls = bundle.targets[:, 2].astype(int)
    feature_names = bundle.feature_names
    proxy_v_idx = feature_names.index("proxy_vmin")
    proxy_l_idx = feature_names.index("proxy_loss_kw")

    models: dict[str, object] = {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=220, min_samples_leaf=2, max_features=0.8,
            random_state=seed, n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=260, min_samples_leaf=1, max_features=0.9,
            random_state=seed, n_jobs=1,
        ),
        "MLP": Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(96, 48), activation="relu", alpha=1e-4,
                max_iter=180, early_stopping=True, random_state=seed,
            )),
        ]),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective="reg:squarederror", random_state=seed, n_jobs=1,
                verbosity=0,
            ),
            n_jobs=1,
        ),
        "LightGBM": MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=300, num_leaves=31, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=seed, n_jobs=1, verbosity=-1,
            ),
            n_jobs=1,
        ),
    }
    classifiers: dict[str, object] = {
        "Ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=400, class_weight="balanced", random_state=seed,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=220, min_samples_leaf=2, max_features=0.8,
            class_weight="balanced", random_state=seed, n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=220, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=1,
        ),
        "MLP": Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(96, 48), activation="relu", alpha=1e-4,
                max_iter=180, early_stopping=True, random_state=seed,
            )),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            class_weight="balanced", random_state=seed, n_jobs=1, verbosity=-1,
        ),
    }

    rows: list[dict[str, float | str | int]] = []
    prediction_store: dict[str, np.ndarray] = {}
    prob_store: dict[str, np.ndarray] = {}

    for split in ["selection", "calibration", "test", "external"]:
        ii = idx[split]
        proxy_pred = np.column_stack([x[ii, proxy_v_idx], x[ii, proxy_l_idx]])
        proxy_prob = ((proxy_pred[:, 0] >= 0.95) & (proxy_pred[:, 0] <= 1.05)).astype(float)
        metrics = prediction_metrics(bundle.targets[ii], proxy_pred, proxy_prob)
        rows.append({"model": "LinearDistFlow", "split": split, "n": len(ii), **metrics, "train_time_s": 0.0, "inference_ms_per_sample": 0.0})
        prediction_store[f"LinearDistFlow__{split}"] = proxy_pred
        prob_store[f"LinearDistFlow__{split}"] = proxy_prob

    for name, model in models.items():
        start = perf_counter()
        model.fit(x[idx["train"]], y_reg[idx["train"]])
        train_time = perf_counter() - start
        classifier = classifiers[name]
        classifier.fit(x[idx["train"]], y_cls[idx["train"]])
        joblib.dump(model, output_dir / f"baseline_{name}_reg.joblib")
        joblib.dump(classifier, output_dir / f"baseline_{name}_cls.joblib")
        for split in ["selection", "calibration", "test", "external"]:
            ii = idx[split]
            start = perf_counter()
            pred_scaled = model.predict(x[ii])
            elapsed = perf_counter() - start
            pred = np.column_stack([pred_scaled[:, 0], np.expm1(pred_scaled[:, 1])])
            prob = classifier.predict_proba(x[ii])[:, 1]
            metrics = prediction_metrics(bundle.targets[ii], pred, prob)
            rows.append({
                "model": name,
                "split": split,
                "n": len(ii),
                **metrics,
                "train_time_s": train_time,
                "inference_ms_per_sample": 1000.0 * elapsed / max(len(ii), 1),
            })
            prediction_store[f"{name}__{split}"] = pred
            prob_store[f"{name}__{split}"] = prob

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "baseline_metrics.csv", index=False)
    np.savez_compressed(output_dir / "baseline_predictions.npz", **prediction_store, **{f"prob__{k}": v for k, v in prob_store.items()})
    return result


class GridDataset(Dataset):
    def __init__(self, bundle: DatasetBundle, indices: np.ndarray, tab_mean: np.ndarray, tab_std: np.ndarray):
        self.node = torch.from_numpy(bundle.node_features[indices])
        self.adj = torch.from_numpy(bundle.adjacency[indices])
        self.mask = torch.from_numpy(bundle.node_mask[indices])
        self.tab = torch.from_numpy((bundle.tabular[indices] - tab_mean) / tab_std)
        self.target = torch.from_numpy(bundle.targets[indices])
        self.indices = torch.from_numpy(indices.astype(np.int64))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.node[i], self.adj[i], self.mask[i], self.tab[i], self.target[i], self.indices[i]


class GraphOutcomeModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        tab_dim: int,
        hidden: int = 40,
        use_graph: bool = True,
        use_reliability: bool = True,
        use_physics_residual: bool = True,
        use_multitask: bool = True,
        simple_fusion: bool = False,
        proxy_v_index: int = 11,
        proxy_loss_index: int = 12,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.use_reliability = use_reliability
        self.use_physics_residual = use_physics_residual
        self.use_multitask = use_multitask
        self.simple_fusion = simple_fusion
        self.proxy_v_index = proxy_v_index
        self.proxy_loss_index = proxy_loss_index
        self.node_in = nn.Linear(node_dim, hidden)
        self.node_h = nn.Linear(hidden, hidden)
        self.reliability = nn.Sequential(nn.Linear(2, 12), nn.ReLU(), nn.Linear(12, 1), nn.Sigmoid())
        fusion_dim = hidden * 2 + tab_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.adaptive_gate = nn.Sequential(nn.Linear(tab_dim, 24), nn.ReLU(), nn.Linear(24, 48), nn.Sigmoid())
        self.reg_head = nn.Linear(48, 2)
        self.cls_head = nn.Linear(48, 1)

    @staticmethod
    def normalized_adjacency(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, n, _ = adj.shape
        eye = torch.eye(n, device=adj.device).unsqueeze(0).expand(batch, -1, -1)
        a = adj + eye * mask.unsqueeze(1)
        degree = a.sum(dim=-1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        return inv_sqrt.unsqueeze(-1) * a * inv_sqrt.unsqueeze(-2)

    def forward(self, node: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor, tab: torch.Tensor):
        h = torch.relu(self.node_in(node))
        if self.use_graph:
            a = self.normalized_adjacency(adj, mask)
            h = torch.relu(torch.bmm(a, h))
            h = torch.relu(self.node_h(h))
            h = torch.relu(torch.bmm(a, h))
        else:
            h = torch.relu(self.node_h(h))

        if self.use_reliability:
            rel_inputs = torch.stack([node[..., 6], node[..., 7]], dim=-1)
            rel = self.reliability(rel_inputs).squeeze(-1) * mask
        else:
            rel = node[..., 6] * mask
        weights = rel / rel.sum(dim=1, keepdim=True).clamp_min(1e-5)
        mean_pool = (h * weights.unsqueeze(-1)).sum(dim=1)
        masked_h = h.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        max_pool = masked_h.max(dim=1).values
        fused = self.fusion(torch.cat([mean_pool, max_pool, tab], dim=1))
        if not self.simple_fusion:
            fused = fused * (0.5 + self.adaptive_gate(tab))
        raw = self.reg_head(fused)
        cls_logit = self.cls_head(fused).squeeze(-1)
        return raw, cls_logit, weights


@dataclass
class TrainedGraphModel:
    model: GraphOutcomeModel
    tab_mean: np.ndarray
    tab_std: np.ndarray
    train_history: pd.DataFrame
    config: dict[str, object]


def train_graph_model(
    bundle: DatasetBundle,
    output_path: str | Path,
    config: dict[str, object] | None = None,
    seed: int = 20260922,
    epochs: int = 24,
    batch_size: int = 128,
) -> TrainedGraphModel:
    if config is None:
        config = {}
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = split_indices(bundle)
    tab_mean = bundle.tabular[idx["train"]].mean(axis=0)
    tab_std = bundle.tabular[idx["train"]].std(axis=0)
    tab_std[tab_std < 1e-6] = 1.0
    feature_names = bundle.feature_names
    proxy_v_index = feature_names.index("proxy_vmin")
    proxy_loss_index = feature_names.index("proxy_loss_kw")
    model = GraphOutcomeModel(
        node_dim=bundle.node_features.shape[-1],
        tab_dim=bundle.tabular.shape[-1],
        proxy_v_index=proxy_v_index,
        proxy_loss_index=proxy_loss_index,
        **config,
    )
    train_ds = GridDataset(bundle, idx["train"], tab_mean, tab_std)
    val_ds = GridDataset(bundle, idx["selection"], tab_mean, tab_std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2e-4)
    bce = nn.BCEWithLogitsLoss()
    history: list[dict[str, float | int]] = []
    best_state = None
    best_val = np.inf
    patience = 6
    stalled = 0

    proxy_v_raw = bundle.tabular[:, proxy_v_index]
    proxy_loss_raw = bundle.tabular[:, proxy_loss_index]

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for node, adj, mask, tab, target, original_idx in train_loader:
            raw, cls_logit, _ = model(node, adj, mask, tab)
            oi = original_idx.numpy()
            if model.use_physics_residual:
                target_reg = torch.column_stack([
                    target[:, 0] - torch.from_numpy(proxy_v_raw[oi]),
                    torch.log1p(target[:, 1].clamp_min(0)) - torch.log1p(torch.from_numpy(proxy_loss_raw[oi]).clamp_min(0)),
                ])
                pred_reg = raw
            else:
                target_reg = torch.column_stack([target[:, 0], torch.log1p(target[:, 1].clamp_min(0))])
                pred_reg = raw
            scale = torch.tensor([0.05, 1.0], dtype=torch.float32)
            reg_loss = torch.mean(((pred_reg - target_reg) / scale) ** 2)
            cls_loss = bce(cls_logit, target[:, 2]) if model.use_multitask else torch.tensor(0.0)
            loss = reg_loss + 0.18 * cls_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(target)
            train_count += len(target)

        val_metrics = evaluate_graph_model(model, bundle, idx["selection"], tab_mean, tab_std, batch_size=batch_size)
        val_objective = val_metrics["vmin_rmse"] / 0.02 + val_metrics["loss_rmse_kw"] / 60.0
        history.append({"epoch": epoch, "train_loss": train_loss_sum / max(train_count, 1), "validation_objective": val_objective})
        if val_objective < best_val - 1e-4:
            best_val = val_objective
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    payload = {
        "state_dict": model.state_dict(),
        "tab_mean": tab_mean,
        "tab_std": tab_std,
        "config": {**config, "proxy_v_index": proxy_v_index, "proxy_loss_index": proxy_loss_index},
        "node_dim": bundle.node_features.shape[-1],
        "tab_dim": bundle.tabular.shape[-1],
    }
    torch.save(payload, output_path)
    return TrainedGraphModel(model, tab_mean, tab_std, history_df, config)


def evaluate_graph_model(
    model: GraphOutcomeModel,
    bundle: DatasetBundle,
    indices: np.ndarray,
    tab_mean: np.ndarray,
    tab_std: np.ndarray,
    batch_size: int = 256,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(GridDataset(bundle, indices, tab_mean, tab_std), batch_size=batch_size, shuffle=False)
    preds = []
    probs = []
    truths = []
    proxy_v_idx = bundle.feature_names.index("proxy_vmin")
    proxy_l_idx = bundle.feature_names.index("proxy_loss_kw")
    with torch.no_grad():
        for node, adj, mask, tab, target, original_idx in loader:
            raw, cls_logit, _ = model(node, adj, mask, tab)
            oi = original_idx.numpy()
            if model.use_physics_residual:
                v = torch.from_numpy(bundle.tabular[oi, proxy_v_idx]) + raw[:, 0]
                log_loss = torch.log1p(torch.from_numpy(bundle.tabular[oi, proxy_l_idx]).clamp_min(0)) + raw[:, 1]
            else:
                v = raw[:, 0]
                log_loss = raw[:, 1]
            loss_kw = torch.expm1(log_loss).clamp_min(0)
            preds.append(torch.column_stack([v, loss_kw]).numpy())
            probs.append(torch.sigmoid(cls_logit).numpy())
            truths.append(target.numpy())
    pred = np.vstack(preds)
    prob = np.concatenate(probs)
    truth = np.vstack(truths)
    return prediction_metrics(truth, pred, prob)


def predict_graph_model(
    trained: TrainedGraphModel,
    bundle: DatasetBundle,
    indices: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = trained.model
    model.eval()
    loader = DataLoader(GridDataset(bundle, indices, trained.tab_mean, trained.tab_std), batch_size=batch_size, shuffle=False)
    preds, probs, weights = [], [], []
    proxy_v_idx = bundle.feature_names.index("proxy_vmin")
    proxy_l_idx = bundle.feature_names.index("proxy_loss_kw")
    with torch.no_grad():
        for node, adj, mask, tab, target, original_idx in loader:
            raw, cls_logit, w = model(node, adj, mask, tab)
            oi = original_idx.numpy()
            if model.use_physics_residual:
                v = torch.from_numpy(bundle.tabular[oi, proxy_v_idx]) + raw[:, 0]
                log_loss = torch.log1p(torch.from_numpy(bundle.tabular[oi, proxy_l_idx]).clamp_min(0)) + raw[:, 1]
            else:
                v = raw[:, 0]
                log_loss = raw[:, 1]
            preds.append(torch.column_stack([v, torch.expm1(log_loss).clamp_min(0)]).numpy())
            probs.append(torch.sigmoid(cls_logit).numpy())
            weights.append(w.numpy())
    return np.vstack(preds), np.concatenate(probs), np.vstack(weights)


def graph_model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
