"""Shared numerical primitives for the notebook experiments.

The proposed GraphOutcomeModel is imported unchanged from models.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from collections.abc import Iterator
import os
import threading

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.special import expit, logit
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from torch import nn
from torch.utils.data import DataLoader

from .models import (
    DatasetBundle,
    GraphOutcomeModel,
    GridDataset,
    TrainedGraphModel,
    predict_graph_model,
)

SEED = 20260922
SEEDS = (20260922, 20260923, 20260924, 20260925, 20260926)
FULL_CONFIG = dict(
    hidden=56,
    use_graph=True,
    use_reliability=True,
    use_physics_residual=True,
    use_multitask=True,
    simple_fusion=True,
)
MODEL_ORDER = [
    "LinearDistFlow",
    "Ridge",
    "RandomForest",
    "ExtraTrees",
    "MLP",
    "XGBoost",
    "LightGBM",
    "GCN Direct",
    "Physics-Residual GCN",
    "ACG-MAC Ensemble",
]
TRAINING = dict(
    epochs=30,
    batch_size=128,
    learning_rate=0.0018,
    weight_decay=0.0002,
    auxiliary_weight=0.18,
    patience=6,
)


@dataclass
class ResourceMeasurement:
    elapsed_s: float = 0.0
    peak_rss_mib: float = 0.0
    incremental_rss_mib: float = 0.0


@contextmanager
def measure_resources() -> Iterator[ResourceMeasurement]:
    """Sample process RSS every 10 ms; absolute RSS includes existing arrays."""
    result = ResourceMeasurement()
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    readings = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.01):
            readings.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = perf_counter()
    try:
        yield result
    finally:
        result.elapsed_s = perf_counter() - start
        readings.append(process.memory_info().rss)
        stop.set()
        monitor.join()
        result.peak_rss_mib = max(readings) / 2**20
        result.incremental_rss_mib = max(0, max(readings) - baseline) / 2**20


def tabular_estimators(name: str, seed: int) -> tuple[object, object]:
    """Return fixed-budget baselines; preprocessing remains inside estimators."""
    if name == "Ridge":
        return (
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=800, class_weight="balanced", random_state=seed),
            ),
        )
    if name in {"RandomForest", "ExtraTrees"}:
        reg_class, cls_class = (
            (RandomForestRegressor, RandomForestClassifier)
            if name == "RandomForest"
            else (ExtraTreesRegressor, ExtraTreesClassifier)
        )
        return (
            TransformedTargetRegressor(
                regressor=reg_class(
                    n_estimators=220,
                    min_samples_leaf=2,
                    max_features=0.9,
                    random_state=seed,
                    n_jobs=1,
                ),
                transformer=StandardScaler(),
            ),
            cls_class(
                n_estimators=220,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        )
    if name == "MLP":
        common = dict(
            hidden_layer_sizes=(96, 48),
            max_iter=300,
            early_stopping=False,
            random_state=seed,
            tol=1e-4,
        )
        return (
            TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(), MLPRegressor(**common)),
                transformer=StandardScaler(),
            ),
            make_pipeline(StandardScaler(), MLPClassifier(**common)),
        )
    if name == "XGBoost":
        common = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
        return MultiOutputRegressor(
            XGBRegressor(objective="reg:squarederror", **common)
        ), XGBClassifier(eval_metric="logloss", **common)
    if name == "LightGBM":
        common = dict(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.03,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=1,
            random_state=seed,
            verbosity=-1,
        )
        return MultiOutputRegressor(LGBMRegressor(**common)), LGBMClassifier(
            class_weight="balanced", **common
        )
    raise ValueError(name)


def fit_tabular(
    name: str, bundle: DatasetBundle, indices: np.ndarray, seed: int = SEED
) -> tuple[object, object]:
    """Fit targets in voltage/log-loss units and feasibility classification."""
    regressor, classifier = tabular_estimators(name, seed)
    valid = indices[bundle.metadata.iloc[indices].converged.to_numpy() == 1]
    regression_target = np.column_stack(
        [bundle.targets[valid, 0], np.log1p(bundle.targets[valid, 1])]
    )
    regressor.fit(bundle.tabular[valid], regression_target)
    classifier.fit(bundle.tabular[indices], bundle.targets[indices, 2])
    return regressor, classifier


def predict_tabular(
    estimators: tuple[object, object] | None, bundle: DatasetBundle, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Predict in physical units without using true loads or test labels."""
    if estimators is None:
        voltage = bundle.tabular[indices, bundle.feature_names.index("proxy_vmin")]
        losses = bundle.tabular[indices, bundle.feature_names.index("proxy_loss_kw")]
        probability = ((voltage >= 0.95) & (voltage <= 1.05)).astype(float)
        return np.column_stack([voltage, losses]), probability
    raw = estimators[0].predict(bundle.tabular[indices])
    prediction = np.column_stack([raw[:, 0], np.maximum(0, np.expm1(raw[:, 1]))])
    probability = estimators[1].predict_proba(bundle.tabular[indices])[:, 1]
    assert np.isfinite(prediction).all()
    return prediction, probability


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Physical-unit errors; caller explicitly selects converged states."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    assert np.isfinite(truth).all() and np.isfinite(prediction).all()
    return dict(
        vmin_rmse=mean_squared_error(truth[:, 0], prediction[:, 0]) ** 0.5,
        vmin_mae=mean_absolute_error(truth[:, 0], prediction[:, 0]),
        vmin_r2=r2_score(truth[:, 0], prediction[:, 0]),
        loss_rmse_kw=mean_squared_error(truth[:, 1], prediction[:, 1]) ** 0.5,
        loss_mae_kw=mean_absolute_error(truth[:, 1], prediction[:, 1]),
        loss_r2=r2_score(truth[:, 1], prediction[:, 1]),
    )


def probability_metrics(
    truth: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Discrimination and probability errors with ten fixed calibration bins."""
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    edges = np.linspace(0, 1, 11)
    bins = np.minimum(np.digitize(probability, edges) - 1, 9)
    ece = sum(
        np.mean(bins == group)
        * abs(truth[bins == group].mean() - probability[bins == group].mean())
        for group in range(10)
        if np.any(bins == group)
    )
    slope_model = LogisticRegression(C=1e6, max_iter=1000).fit(
        logit(probability).reshape(-1, 1), truth
    )
    return dict(
        auroc=roc_auc_score(truth, probability),
        auprc=average_precision_score(truth, probability),
        f1=f1_score(truth, probability >= threshold),
        brier=brier_score_loss(truth, probability),
        log_loss=log_loss(truth, probability),
        ece=ece,
        calibration_slope=float(slope_model.coef_[0, 0]),
    )


def calibrate_probability(
    truth: np.ndarray, calibration_probability: np.ndarray, target_probability: np.ndarray
) -> tuple[np.ndarray, float, object]:
    """Fit Platt map and F1 threshold exclusively on the calibration role."""
    features = logit(np.clip(calibration_probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    mapper = LogisticRegression(C=1.0, max_iter=1000).fit(features, truth)
    calibrated = mapper.predict_proba(features)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(truth, calibrated >= threshold) for threshold in thresholds]
    threshold = float(thresholds[int(np.argmax(scores))])
    target_features = logit(np.clip(target_probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return mapper.predict_proba(target_features)[:, 1], threshold, mapper


def train_graph(
    bundle: DatasetBundle,
    train_indices: np.ndarray,
    selection_indices: np.ndarray,
    config: dict,
    seed: int = SEED,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.0018,
    weight_decay: float = 0.0002,
    auxiliary_weight: float = 0.18,
    patience: int = 6,
    scale_voltage: float = 0.05,
    scale_log_loss: float = 1.0,
    disable_dropout: bool = False,
) -> TrainedGraphModel:
    """Train the unchanged graph architecture on explicit disjoint groups."""
    assert not np.intersect1d(train_indices, selection_indices).size
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    mean = bundle.tabular[train_indices].mean(axis=0)
    std = bundle.tabular[train_indices].std(axis=0)
    std[std < 1e-6] = 1.0
    model = GraphOutcomeModel(
        bundle.node_features.shape[-1], bundle.tabular.shape[-1], **config
    )
    if disable_dropout:
        for layer in model.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = 0.0
    loader = DataLoader(
        GridDataset(bundle, train_indices, mean, std), batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    voltage_index = bundle.feature_names.index("proxy_vmin")
    loss_index = bundle.feature_names.index("proxy_loss_kw")
    best_score, stalled, best_state = np.inf, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for node, adjacency, mask, tabular, target, original_indices in loader:
            raw, logits, _ = model(node, adjacency, mask, tabular)
            rows = original_indices.numpy()
            regression_target = torch.column_stack(
                [target[:, 0], torch.log1p(target[:, 1].clamp_min(0))]
            )
            if model.use_physics_residual:
                proxy = torch.from_numpy(
                    np.column_stack(
                        [
                            bundle.tabular[rows, voltage_index],
                            np.log1p(bundle.tabular[rows, loss_index].clip(min=0)),
                        ]
                    )
                )
                regression_target = regression_target - proxy
            valid = torch.from_numpy(bundle.metadata.iloc[rows].converged.to_numpy() == 1)
            scales = torch.tensor([scale_voltage, scale_log_loss])
            regression_loss = (((raw[valid] - regression_target[valid]) / scales) ** 2).mean()
            classification_loss = (
                nn.functional.binary_cross_entropy_with_logits(logits, target[:, 2])
                if model.use_multitask
                else logits.sum() * 0
            )
            objective = regression_loss + auxiliary_weight * classification_loss
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(objective.item()) * len(rows)
        trained = TrainedGraphModel(model, mean, std, pd.DataFrame(), config)
        predictions, _, _ = predict_graph_model(trained, bundle, selection_indices)
        valid_selection = bundle.metadata.iloc[selection_indices].converged.to_numpy() == 1
        metrics = regression_metrics(
            bundle.targets[selection_indices][valid_selection], predictions[valid_selection]
        )
        score = metrics["vmin_rmse"] / 0.02 + metrics["loss_rmse_kw"] / 60.0
        history.append(
            dict(
                epoch=epoch,
                train_objective=loss_sum / len(train_indices),
                selection_objective=score,
            )
        )
        if score < best_score - 1e-4:
            best_score, stalled = score, 0
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        else:
            stalled += 1
            if stalled >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return TrainedGraphModel(model, mean, std, pd.DataFrame(history), config.copy())


def save_graph(trained: TrainedGraphModel, path: Path, bundle: DatasetBundle) -> None:
    """Save architecture, fitted preprocessing and weights together."""
    torch.save(
        dict(
            state_dict=trained.model.state_dict(),
            tab_mean=trained.tab_mean,
            tab_std=trained.tab_std,
            config=trained.config,
            node_dim=bundle.node_features.shape[-1],
            tab_dim=bundle.tabular.shape[-1],
        ),
        path,
    )


def inner_group_selection(
    bundle: DatasetBundle, indices: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve whole fold-training topologies for early stopping."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    fit, select = next(
        splitter.split(indices, groups=bundle.metadata.iloc[indices].topology_id)
    )
    return indices[fit], indices[select]


def cluster_bootstrap(
    delta: np.ndarray, groups: np.ndarray, seed: int = SEED, draws: int = 2000
) -> tuple[float, float, float]:
    """Resample independent topologies, retaining all their observations."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    sums = np.array([delta[groups == group].sum() for group in unique])
    counts = np.array([(groups == group).sum() for group in unique])
    sampled = rng.integers(0, len(unique), (draws, len(unique)))
    means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return (
        float(delta.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Control familywise error across the declared paired comparisons."""
    order = np.argsort(pvalues)
    adjusted = np.maximum.accumulate(
        np.asarray(pvalues)[order] * (len(pvalues) - np.arange(len(pvalues)))
    )
    result = np.empty(len(pvalues))
    result[order] = np.minimum(1, adjusted)
    return result
