#!/usr/bin/env python3

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis_outputs"
FIG_DIR = OUT_DIR / "figs"
TABLE_DIR = OUT_DIR / "tables"
TEXT_DIR = OUT_DIR / "text"
PAPER_FIG_DIR = ROOT / "figs"

MAIN_PREFIXES = ("gpt-oss-bio", "gpt-oss-mat", "qwen-bio")
GROUP_PREFIX = "group_"
POW2 = [1, 2, 4, 8, 16, 32, 64, 128]
WIDTH_TICKS = [1, 2, 4, 8, 16, 32, 64, 128]
MAIN_MODEL = "gpt-oss-20b"
HIGH_BUDGET = 128
MODEL_MAP = {
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "openai/Qwen3-30B-A3B-Thinking-2507": "qwen3-30b-a3b",
}
MODEL_LABELS = {
    "gpt-oss-20b": "gpt-oss-20b",
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
}
DOMAIN_LABELS = {
    "bio": "Bio",
    "material": "Material",
}
ALGORITHM_LABELS = {
    "pbeam": "PBeam",
    "pie": "PIE",
    "openevolve": "OpenEvolve",
}
PALETTE = {
    1: "#19124D",
    2: "#1A50A1",
    4: "#00869C",
    8: "#1EB15D",
    16: "#93CA26",
    32: "#F3BA00",
    64: "#EB5D04",
    128: "#BE1525",
}
MODEL_COLORS = {
    "gpt-oss-20b": "#1f77b4",
    "qwen3-30b-a3b": "#d62728",
}
FIT_COLORS = {
    "log_power_budget": "#d62728",
    "anchored_log_power_budget": "#ff7f0e",
    "exp_log_power_budget": "#1f77b4",
    "hinge_log_power_budget": "#2ca02c",
    "hinge_exp_log_power_budget": "#17becf",
    "cubic_logwidth_budget": "#9467bd",
    "hinge_cubic_logwidth_budget": "#7f3c8d",
    "shifted_log_power_budget": "#8c564b",
    "power_law": "#2ca02c",
    "sqrt_law": "#8c564b",
    "gap_power_law": "#2ca02c",
    "log_linear_acc": "#ff7f0e",
}
ALLOWED_MODEL_DOMAIN = {
    ("gpt-oss-20b", "bio"),
    ("gpt-oss-20b", "material"),
    ("qwen3-30b-a3b", "bio"),
}


@dataclass
class FitResult:
    name: str
    params: dict[str, float]
    yhat: np.ndarray
    r2: float
    mae: float
    rmse: float
    fit_min_budget: int = 1


def ensure_dirs() -> None:
    for path in (OUT_DIR, FIG_DIR, TABLE_DIR, TEXT_DIR, PAPER_FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)


def save_figure(fig: plt.Figure, basename: str) -> None:
    for fig_dir in (FIG_DIR, PAPER_FIG_DIR):
        fig.savefig(fig_dir / f"{basename}.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / f"{basename}.png", dpi=240, bbox_inches="tight")


def list_files(prefixes: tuple[str, ...] | str) -> list[Path]:
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    return sorted(
        path
        for path in ROOT.iterdir()
        if path.is_file() and any(path.name.startswith(prefix) for prefix in prefixes)
    )


def preprocess(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    df = df.copy()
    df["source_type"] = source_type
    df["domain"] = np.where(df["task"].str.contains("bio"), "bio", "material")
    df["model_short"] = df["model"].map(MODEL_MAP).fillna(df["model"])
    df["width"] = (df["B"] * df["K"]).astype(int)
    df["compute"] = (df["width"] * df["iteration"]).astype(int)
    df["budget_pow2"] = df["compute"].isin(POW2)
    df["beam_fraction"] = df["B"] / df["width"]
    df["beam_to_sample_ratio"] = df["B"] / df["K"]
    df["per_group_width"] = (df["width"] / df["G"]).astype(float)
    df["is_greedy"] = (df["B"] == 1) & (df["K"] == 1)
    df["is_single_iter"] = df["iteration"] == 1
    return df


def load_main_results() -> pd.DataFrame:
    files = list_files(MAIN_PREFIXES)
    frames = [preprocess(pd.read_csv(path).assign(source_file=path.name), "main") for path in files]
    df = pd.concat(frames, ignore_index=True)
    keep_mask = df[["model_short", "domain"]].apply(tuple, axis=1).isin(ALLOWED_MODEL_DOMAIN)
    return df[keep_mask].reset_index(drop=True)


def load_group_results() -> pd.DataFrame:
    files = list_files(GROUP_PREFIX)
    frames = [preprocess(pd.read_csv(path).assign(source_file=path.name), "group") for path in files]
    return pd.concat(frames, ignore_index=True)


def fill_theoretical_pie_equivalents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_imputed"] = False
    key_cols = ["model_short", "task", "seed", "B", "K", "G", "iteration"]
    pie_keys = set(tuple(row) for row in df[df["algorithm"] == "pie"][key_cols].to_numpy())
    candidates = df[(df["algorithm"] == "pbeam") & (df["is_greedy"] | df["is_single_iter"])]
    rows = []
    for _, row in candidates.iterrows():
        key = tuple(row[col] for col in key_cols)
        if key in pie_keys:
            continue
        new_row = row.copy()
        new_row["algorithm"] = "pie"
        new_row["is_imputed"] = True
        rows.append(new_row)
    if not rows:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def domain_seed_average(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_short",
        "domain",
        "algorithm",
        "seed",
        "B",
        "K",
        "G",
        "width",
        "compute",
        "iteration",
        "beam_fraction",
        "beam_to_sample_ratio",
        "per_group_width",
    ]
    agg_cols = ["train_acc01", "test_acc01", "wall_time_s"]
    return (
        df.groupby(group_cols, as_index=False)[agg_cols]
        .mean()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def summarize_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_short",
        "domain",
        "algorithm",
        "B",
        "K",
        "G",
        "width",
        "compute",
        "iteration",
        "beam_fraction",
        "beam_to_sample_ratio",
        "per_group_width",
    ]
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            train_acc01_mean=("train_acc01", "mean"),
            train_acc01_std=("train_acc01", "std"),
            test_acc01_mean=("test_acc01", "mean"),
            test_acc01_std=("test_acc01", "std"),
            wall_time_s_mean=("wall_time_s", "mean"),
            wall_time_s_std=("wall_time_s", "std"),
            num_seeds=("seed", "nunique"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    for col in ("train_acc01_std", "test_acc01_std", "wall_time_s_std"):
        summary[col] = summary[col].fillna(0.0)
    return summary


def choose_best_config(
    summary_df: pd.DataFrame,
    group_cols: list[str],
    metric: str = "train_acc01_mean",
) -> pd.DataFrame:
    ordered = summary_df.sort_values(
        group_cols + [metric, "test_acc01_mean", "wall_time_s_mean", "B", "K"],
        ascending=[True] * len(group_cols) + [False, False, True, True, True],
    )
    best = ordered.groupby(group_cols, as_index=False).first()
    return best.reset_index(drop=True)


def choose_smallest_within_tolerance(
    summary_df: pd.DataFrame,
    group_cols: list[str],
    metric: str = "train_acc01_mean",
    tolerance_abs: float = 0.001,
) -> pd.DataFrame:
    rows = []
    for _, sub in summary_df.groupby(group_cols, sort=True):
        sub = sub.copy()
        best_metric = sub[metric].max()
        eligible = sub[sub[metric] >= best_metric - tolerance_abs]
        pick = eligible.sort_values(
            ["width", "wall_time_s_mean", "test_acc01_mean", "B", "K"],
            ascending=[True, True, False, True, True],
        ).iloc[0]
        rows.append(pick)
    return pd.DataFrame(rows).reset_index(drop=True)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.sum((y - y.mean()) ** 2)
    if math.isclose(denom, 0.0):
        return 1.0
    return 1.0 - np.sum((y - yhat) ** 2) / denom


def mae_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def rmse_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def power_law_model(x: np.ndarray, c: float, alpha: float) -> np.ndarray:
    return c * np.power(x, alpha)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    logx = np.log2(x)
    logy = np.log2(y)
    slope, intercept = np.polyfit(logx, logy, 1)
    yhat = power_law_model(x, 2 ** intercept, slope)
    return FitResult(
        name="power_law",
        params={"c": float(2 ** intercept), "alpha": float(slope)},
        yhat=yhat,
        r2=r2_score(y, yhat),
        mae=mae_score(y, yhat),
        rmse=rmse_score(y, yhat),
    )


def sqrt_law_model(x: np.ndarray, c: float) -> np.ndarray:
    return c * np.sqrt(x)


def fit_sqrt_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    denom = np.sqrt(x)
    c = float(np.dot(y, denom) / np.dot(denom, denom))
    yhat = sqrt_law_model(x, c)
    return FitResult(
        name="sqrt_law",
        params={"c": c},
        yhat=yhat,
        r2=r2_score(y, yhat),
        mae=mae_score(y, yhat),
        rmse=rmse_score(y, yhat),
    )


def log_power_width_model(x: np.ndarray, c: float, alpha: float) -> np.ndarray:
    return c * np.power(np.log2(x), alpha)


def fit_log_power_width_law(x: np.ndarray, y: np.ndarray, fit_min_budget: int = 2) -> FitResult:
    mask = x >= fit_min_budget
    fit_x = x[mask]
    fit_y = y[mask]
    p0 = [max(0.01, float(np.max(fit_y)) / max(np.log2(np.max(fit_x)) ** 2, 1.0)), 2.0]
    bounds = ([1e-8, 0.05], [100.0, 8.0])
    popt, _ = curve_fit(log_power_width_model, fit_x, fit_y, p0=p0, bounds=bounds, maxfev=50000)
    yhat = np.full_like(x, np.nan, dtype=float)
    yhat[mask] = log_power_width_model(fit_x, *popt)
    return FitResult(
        name="log_power_budget",
        params={"c": float(popt[0]), "alpha": float(popt[1])},
        yhat=yhat,
        r2=r2_score(fit_y, yhat[mask]),
        mae=mae_score(fit_y, yhat[mask]),
        rmse=rmse_score(fit_y, yhat[mask]),
        fit_min_budget=fit_min_budget,
    )


def anchored_log_power_width_model(x: np.ndarray, c: float, alpha: float) -> np.ndarray:
    return 1.0 + c * np.power(np.log2(x), alpha)


def fit_anchored_log_power_width_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    p0 = [0.01, 4.0]
    bounds = ([1e-8, 0.05], [100.0, 10.0])
    popt, _ = curve_fit(
        anchored_log_power_width_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    yhat = anchored_log_power_width_model(x, *popt)
    return FitResult(
        name="anchored_log_power_budget",
        params={"c": float(popt[0]), "alpha": float(popt[1])},
        yhat=yhat,
        r2=r2_score(y, yhat),
        mae=mae_score(y, yhat),
        rmse=rmse_score(y, yhat),
    )


def exp_log_power_width_model(x: np.ndarray, c: float, alpha: float) -> np.ndarray:
    return np.exp(c * np.power(np.log2(x), alpha))


def fit_exp_log_power_width_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    p0 = [0.1, 1.5]
    bounds = ([1e-8, 0.05], [5.0, 8.0])
    popt, _ = curve_fit(
        exp_log_power_width_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    yhat = exp_log_power_width_model(x, *popt)
    return FitResult(
        name="exp_log_power_budget",
        params={"c": float(popt[0]), "alpha": float(popt[1])},
        yhat=yhat,
        r2=r2_score(y, yhat),
        mae=mae_score(y, yhat),
        rmse=rmse_score(y, yhat),
    )


def hinge_log_power_width_model(x: np.ndarray, c: float, alpha: float) -> np.ndarray:
    s = np.maximum(np.log2(x) - 1.0, 0.0)
    return 1.0 + c * np.power(s, alpha)


def fit_hinge_log_power_width_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    p0 = [0.05, 3.0]
    bounds = ([1e-8, 0.05], [100.0, 10.0])
    popt, _ = curve_fit(
        hinge_log_power_width_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    yhat = hinge_log_power_width_model(x, *popt)
    return FitResult(
        name="hinge_log_power_budget",
        params={"c": float(popt[0]), "alpha": float(popt[1])},
        yhat=yhat,
        r2=r2_score(y, yhat),
        mae=mae_score(y, yhat),
        rmse=rmse_score(y, yhat),
    )


def hinge_exp_log_power_width_model(x: np.ndarray, c: float, alpha: float) -> np.ndarray:
    s = np.maximum(np.log2(x) - 1.0, 0.0)
    return np.exp(c * np.power(s, alpha))


def fit_hinge_exp_log_power_width_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    p0 = [0.2, 1.5]
    bounds = ([1e-8, 0.05], [5.0, 8.0])
    popt, _ = curve_fit(
        hinge_exp_log_power_width_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    yhat = hinge_exp_log_power_width_model(x, *popt)
    return FitResult(
        name="hinge_exp_log_power_budget",
        params={"c": float(popt[0]), "alpha": float(popt[1])},
        yhat=yhat,
        r2=r2_score(y, yhat),
        mae=mae_score(y, yhat),
        rmse=rmse_score(y, yhat),
    )


def shifted_log_power_width_model(x: np.ndarray, c: float, alpha: float, bias: float) -> np.ndarray:
    return bias + c * np.power(np.log2(x), alpha)


def fit_shifted_log_power_width_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    p0 = [0.05, 2.0, max(0.0, float(np.min(y) - 0.25))]
    bounds = ([1e-8, 0.05, 0.0], [100.0, 8.0, 16.0])
    popt, _ = curve_fit(
        shifted_log_power_width_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    yhat = shifted_log_power_width_model(x, *popt)
    return FitResult(
        name="shifted_log_power_budget",
        params={"c": float(popt[0]), "alpha": float(popt[1]), "bias": float(popt[2])},
        yhat=yhat,
        r2=r2_score(y, yhat),
        mae=mae_score(y, yhat),
        rmse=rmse_score(y, yhat),
    )


def cubic_logwidth_width_model(x: np.ndarray, a1: float, a2: float, a3: float) -> np.ndarray:
    l = np.log2(x)
    return np.exp(a1 * l + a2 * (l ** 2) + a3 * (l ** 3))


def fit_cubic_logwidth_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    p0 = [0.0, 0.05, 0.001]
    bounds = ([-2.0, -2.0, -2.0], [4.0, 4.0, 2.0])
    popt, _ = curve_fit(
        cubic_logwidth_width_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=200000,
    )
    yhat = cubic_logwidth_width_model(x, *popt)
    return FitResult(
        name="cubic_logwidth_budget",
        params={"a1": float(popt[0]), "a2": float(popt[1]), "a3": float(popt[2])},
        yhat=yhat,
        r2=r2_score(y, yhat),
        mae=mae_score(y, yhat),
        rmse=rmse_score(y, yhat),
    )


def hinge_cubic_logwidth_width_model(x: np.ndarray, a1: float, a2: float, a3: float) -> np.ndarray:
    s = np.maximum(np.log2(x) - 1.0, 0.0)
    return np.exp(a1 * s + a2 * (s ** 2) + a3 * (s ** 3))


def fit_hinge_cubic_logwidth_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    p0 = [0.4, 0.0, 0.001]
    bounds = ([-4.0, -4.0, -4.0], [4.0, 4.0, 4.0])
    popt, _ = curve_fit(
        hinge_cubic_logwidth_width_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=200000,
    )
    yhat = hinge_cubic_logwidth_width_model(x, *popt)
    return FitResult(
        name="hinge_cubic_logwidth_budget",
        params={"a1": float(popt[0]), "a2": float(popt[1]), "a3": float(popt[2])},
        yhat=yhat,
        r2=r2_score(y, yhat),
        mae=mae_score(y, yhat),
        rmse=rmse_score(y, yhat),
    )


def power_gap_model(x: np.ndarray, c: float, alpha: float) -> np.ndarray:
    return c * np.power(x, -alpha)


def fit_gap_power_law(x: np.ndarray, acc: np.ndarray) -> FitResult:
    gap = np.maximum(1e-6, 1.0 - acc)
    p0 = [float(gap.max()), 0.5]
    bounds = ([1e-8, 0.01], [10.0, 4.0])
    popt, _ = curve_fit(power_gap_model, x, gap, p0=p0, bounds=bounds, maxfev=50000)
    gap_hat = power_gap_model(x, *popt)
    yhat = 1.0 - gap_hat
    return FitResult(
        name="gap_power_law",
        params={"c": float(popt[0]), "alpha": float(popt[1])},
        yhat=yhat,
        r2=r2_score(acc, yhat),
        mae=mae_score(acc, yhat),
        rmse=rmse_score(acc, yhat),
    )


def log_linear_acc_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(a + b * np.log2(x), 0.0, 1.0)


def fit_log_linear_acc(x: np.ndarray, acc: np.ndarray) -> FitResult:
    p0 = [float(acc.min()), float((acc.max() - acc.min()) / max(np.ptp(np.log2(x)), 1.0))]
    popt, _ = curve_fit(log_linear_acc_model, x, acc, p0=p0, maxfev=50000)
    yhat = log_linear_acc_model(x, *popt)
    return FitResult(
        name="log_linear_acc",
        params={"a": float(popt[0]), "b": float(popt[1])},
        yhat=yhat,
        r2=r2_score(acc, yhat),
        mae=mae_score(acc, yhat),
        rmse=rmse_score(acc, yhat),
    )


def add_frontier_flag(df: pd.DataFrame, time_col: str, perf_col: str) -> pd.DataFrame:
    df = df.sort_values([time_col, perf_col], ascending=[True, False]).copy()
    best_so_far = -np.inf
    frontier = []
    for _, row in df.iterrows():
        if row[perf_col] > best_so_far + 1e-12:
            frontier.append(True)
            best_so_far = row[perf_col]
        else:
            frontier.append(False)
    df["is_pareto"] = frontier
    return df


def pretty_number(value: float) -> str:
    if abs(value) >= 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2e}"


def format_width_fit_text(fit_row: pd.Series) -> str:
    score_text = rf"$R^2={float(fit_row['r2']):.3f}$, $\mathrm{{MAE}}={float(fit_row['mae']):.2f}$"
    if fit_row["fit_name"] == "hinge_log_power_budget":
        c = pretty_number(float(fit_row["param_a"]))
        alpha = f"{float(fit_row['param_alpha']):.2f}"
        return (
            rf"$w^*=1+{c}\,[\log_2 N-1]_+^{{{alpha}}}$"
            "\n"
            f"{score_text}"
        )
    if fit_row["fit_name"] == "hinge_cubic_logwidth_budget":
        a1 = pretty_number(float(fit_row["param_a"]))
        a2 = pretty_number(float(fit_row["param_alpha"]))
        a3 = pretty_number(float(fit_row["param_bias"]))
        return (
            r"$w^*= \exp(a_1s+a_2s^2+a_3s^3)$"
            "\n"
            r"$s=[\log_2 N-1]_+$"
            "\n"
            rf"$(a_1,a_2,a_3)=({a1},{a2},{a3})$"
            "\n"
            f"{score_text}"
        )
    if fit_row["fit_name"] == "cubic_logwidth_budget":
        a1 = pretty_number(float(fit_row["param_a"]))
        a2 = pretty_number(float(fit_row["param_alpha"]))
        a3 = pretty_number(float(fit_row["param_bias"]))
        return (
            r"$w^*= \exp(a_1L+a_2L^2+a_3L^3)$"
            "\n"
            rf"$L=\log_2 N$, $(a_1,a_2,a_3)=({a1},{a2},{a3})$"
            "\n"
            f"{score_text}"
        )
    if fit_row["fit_name"] == "hinge_exp_log_power_budget":
        c = pretty_number(float(fit_row["param_a"]))
        alpha = f"{float(fit_row['param_alpha']):.2f}"
        return (
            rf"$w^*= \exp({c}\,[\log_2 N-1]_+^{{{alpha}}})$"
            "\n"
            f"{score_text}"
        )
    c = pretty_number(float(fit_row["param_a"]))
    alpha = f"{float(fit_row['param_alpha']):.2f}"
    return f"$w^*= {c}(\\log_2 N)^{{{alpha}}}$\n{score_text}"


def collect_unique_legend_items(ax_list: list[plt.Axes]) -> tuple[list[object], list[str]]:
    seen: set[str] = set()
    handles: list[object] = []
    labels: list[str] = []
    for ax in ax_list:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label in seen or label.startswith("_"):
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def plot_width_scaling_main(
    width_frontier: pd.DataFrame,
    width_optima: pd.DataFrame,
    width_optima_tol: pd.DataFrame,
    fit_table: pd.DataFrame,
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10.2),
        sharex="col",
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    panel_order = [
        ("gpt-oss-20b", "bio"),
        ("gpt-oss-20b", "material"),
        ("qwen3-30b-a3b", "bio"),
    ]
    for col, (model, domain) in enumerate(panel_order):
        frontier = width_frontier[
            (width_frontier["model_short"] == model)
            & (width_frontier["domain"] == domain)
        ].copy()
        opt = width_optima[
            (width_optima["model_short"] == model)
            & (width_optima["domain"] == domain)
        ].copy()
        opt_tol = width_optima_tol[
            (width_optima_tol["model_short"] == model)
            & (width_optima_tol["domain"] == domain)
        ].copy()
        fit_row = fit_table[
            (fit_table["model_short"] == model)
            & (fit_table["domain"] == domain)
            & (fit_table["fit_name"] == "hinge_log_power_budget")
        ].iloc[0]

        ax_top = axes[0, col]
        for width in sorted(frontier["width"].unique()):
            sub = frontier[frontier["width"] == width].sort_values("compute")
            ax_top.plot(
                sub["compute"],
                sub["train_acc01_mean"],
                marker="o",
                markersize=4.5,
                linewidth=1.9,
                color=PALETTE.get(int(width), "#999999"),
                alpha=0.95,
                label=f"$w={int(width)}$",
            )
        ax_top.plot(
            opt["compute"],
            opt["train_acc01_mean"],
            color="black",
            linewidth=1.1,
            linestyle=":",
            alpha=0.8,
        )
        ax_top.scatter(
            opt["compute"],
            opt["train_acc01_mean"],
            c=[PALETTE.get(int(w), "#999999") for w in opt["width"]],
            edgecolors="black",
            linewidths=0.5,
            marker="*",
            s=170,
            zorder=6,
            label="Best over PBeam/PIE",
        )
        ax_top.set_xscale("log", base=2)
        ax_top.set_xticks(POW2)
        ax_top.set_title(f"{MODEL_LABELS[model]} | {DOMAIN_LABELS[domain]}")
        ax_top.set_ylabel(r"Mean train $\mathrm{Acc}_{0.1}$")
        ax_top.set_xlabel(r"Budget $N = nkT$")

        ax_bottom = axes[1, col]
        ax_bottom.plot(
            opt["compute"],
            opt["width"],
            color="black",
            marker="o",
            linewidth=1.8,
            label="Exact optimum",
        )
        ax_bottom.plot(
            opt_tol["compute"],
            opt_tol["width"],
            color="#666666",
            marker="s",
            linewidth=1.6,
            linestyle="--",
            label="Smallest within 0.1 pp",
        )
        fit_start = int(fit_row["fit_min_budget"])
        xs = np.geomspace(fit_start, max(POW2), 256)
        ys = width_fit_predict(fit_row, xs)
        ax_bottom.plot(
            xs,
            ys,
            color=FIT_COLORS["hinge_log_power_budget"],
            linewidth=2.1,
            label=r"Fit: $1 + c[\log_2 N-1]_+^\alpha$",
        )
        ax_bottom.set_xscale("log", base=2)
        ax_bottom.set_xticks(POW2)
        ax_bottom.set_yscale("log", base=2)
        ax_bottom.set_yticks(WIDTH_TICKS)
        ax_bottom.set_ylabel(r"Optimal width $w^*$")
        ax_bottom.set_xlabel(r"Budget $N$")
        ax_bottom.text(
            0.04,
            0.94,
            format_width_fit_text(fit_row),
            transform=ax_bottom.transAxes,
            va="top",
            fontsize=10.5,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#dddddd"},
        )

    handles, labels = collect_unique_legend_items(list(axes.flatten()))
    fig.legend(handles, labels, ncol=7, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig_width_scaling_main")
    plt.close(fig)


def draw_node(ax: plt.Axes, x: float, y: float, label: str = "", color: str = "#4c78a8", radius: float = 0.045, alpha: float = 1.0) -> None:
    node = Circle((x, y), radius=radius, facecolor=color, edgecolor="white", linewidth=1.1, alpha=alpha)
    ax.add_patch(node)
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color="white", weight="bold")


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#666666", lw: float = 1.6, alpha: float = 1.0, style: str = "->") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=10,
        shrinkB=10,
    )
    ax.add_patch(arrow)


def setup_schematic_axis(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10, weight="bold")
    ax.text(0.5, -0.08, subtitle, ha="center", va="top", fontsize=10)


def plot_width_frontier_all(width_frontier: pd.DataFrame, width_optima: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), sharex=True, sharey=False)
    panel_order = [
        ("gpt-oss-20b", "bio"),
        ("gpt-oss-20b", "material"),
        ("qwen3-30b-a3b", "bio"),
    ]
    flat_axes = np.atleast_1d(axes)
    for ax, (model, domain) in zip(flat_axes, panel_order):
        sub = width_frontier[
            (width_frontier["model_short"] == model)
            & (width_frontier["domain"] == domain)
        ].copy()
        opt = width_optima[
            (width_optima["model_short"] == model)
            & (width_optima["domain"] == domain)
        ].copy()
        for width in sorted(sub["width"].unique()):
            width_df = sub[sub["width"] == width].sort_values("compute")
            ax.plot(
                width_df["compute"],
                width_df["train_acc01_mean"],
                marker="o",
                linewidth=1.8,
                markersize=4.2,
                color=PALETTE.get(int(width), "#999999"),
                alpha=0.92,
                label=f"$w={int(width)}$",
            )
        ax.scatter(
            opt["compute"],
            opt["train_acc01_mean"],
            c=[PALETTE.get(int(w), "#999999") for w in opt["width"]],
            edgecolors="black",
            linewidths=0.5,
            marker="*",
            s=150,
            zorder=5,
            label="Best over PBeam/PIE",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(POW2)
        ax.set_title(f"{MODEL_LABELS[model]} | {DOMAIN_LABELS[domain]}")
        ax.set_xlabel(r"Budget $N = nkT$")
        ax.set_ylabel(r"Mean train $\mathrm{Acc}_{0.1}$")
    handles, labels = collect_unique_legend_items(list(flat_axes))
    fig.legend(handles, labels, ncol=5, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_figure(fig, "fig_width_frontier_all")
    plt.close(fig)


def plot_width_frontier_main(width_frontier: pd.DataFrame, width_optima: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.95), sharex=True, sharey=False)
    panel_order = [
        ("gpt-oss-20b", "bio"),
        ("gpt-oss-20b", "material"),
        ("qwen3-30b-a3b", "bio"),
    ]
    flat_axes = np.atleast_1d(axes)
    for ax, (model, domain) in zip(flat_axes, panel_order):
        sub = width_frontier[
            (width_frontier["model_short"] == model)
            & (width_frontier["domain"] == domain)
        ].copy()
        opt = width_optima[
            (width_optima["model_short"] == model)
            & (width_optima["domain"] == domain)
        ].copy()
        for width in sorted(sub["width"].unique()):
            width_df = sub[sub["width"] == width].sort_values("compute")
            ax.plot(
                width_df["compute"],
                width_df["train_acc01_mean"],
                marker="o",
                linewidth=1.5,
                markersize=3.4,
                color=PALETTE.get(int(width), "#999999"),
                alpha=0.92,
                label=f"$w={int(width)}$",
            )
        ax.scatter(
            opt["compute"],
            opt["train_acc01_mean"],
            c=[PALETTE.get(int(w), "#999999") for w in opt["width"]],
            edgecolors="black",
            linewidths=0.5,
            marker="*",
            s=112,
            zorder=5,
            label="Best over PBeam/PIE",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(POW2)
        ax.set_title(f"{MODEL_LABELS[model]} | {DOMAIN_LABELS[domain]}")
        ax.set_xlabel(r"Budget $N = nkT$")
        ax.set_ylabel(r"Mean train $\mathrm{Acc}_{0.1}$")
    handles, labels = collect_unique_legend_items(list(flat_axes))
    fig.legend(handles, labels, ncol=5, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, "fig_width_frontier_main")
    plt.close(fig)


def width_fit_predict(row: pd.Series, x: np.ndarray) -> np.ndarray:
    fit_name = row["fit_name"]
    if fit_name == "log_power_budget":
        y = np.full_like(x, np.nan, dtype=float)
        mask = x >= float(row["fit_min_budget"])
        y[mask] = log_power_width_model(x[mask], float(row["param_a"]), float(row["param_alpha"]))
        return y
    if fit_name == "anchored_log_power_budget":
        return anchored_log_power_width_model(x, float(row["param_a"]), float(row["param_alpha"]))
    if fit_name == "exp_log_power_budget":
        return exp_log_power_width_model(x, float(row["param_a"]), float(row["param_alpha"]))
    if fit_name == "hinge_log_power_budget":
        return hinge_log_power_width_model(x, float(row["param_a"]), float(row["param_alpha"]))
    if fit_name == "hinge_exp_log_power_budget":
        return hinge_exp_log_power_width_model(x, float(row["param_a"]), float(row["param_alpha"]))
    if fit_name == "cubic_logwidth_budget":
        return cubic_logwidth_width_model(x, float(row["param_a"]), float(row["param_alpha"]), float(row["param_bias"]))
    if fit_name == "hinge_cubic_logwidth_budget":
        return hinge_cubic_logwidth_width_model(x, float(row["param_a"]), float(row["param_alpha"]), float(row["param_bias"]))
    if fit_name == "shifted_log_power_budget":
        return shifted_log_power_width_model(x, float(row["param_a"]), float(row["param_alpha"]), float(row["param_bias"]))
    if fit_name == "power_law":
        return power_law_model(x, float(row["param_a"]), float(row["param_alpha"]))
    if fit_name == "sqrt_law":
        return sqrt_law_model(x, float(row["param_a"]))
    raise ValueError(f"Unsupported fit {fit_name}")


def plot_width_fit_compare(width_optima: pd.DataFrame, fit_table: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharex=True, sharey=False)
    panel_order = [
        ("gpt-oss-20b", "bio"),
        ("gpt-oss-20b", "material"),
        ("qwen3-30b-a3b", "bio"),
    ]
    fit_order = [
        ("log_power_budget", r"$c(\log_2 N)^\alpha$"),
        ("hinge_log_power_budget", r"$1+c[\log_2 N-1]_+^\alpha$"),
        ("hinge_exp_log_power_budget", r"$\exp(c[\log_2 N-1]_+^\alpha)$"),
        ("hinge_cubic_logwidth_budget", r"$\exp(a_1s+a_2s^2+a_3s^3)$"),
    ]
    for ax, (model, domain) in zip(axes, panel_order):
        opt = width_optima[
            (width_optima["model_short"] == model)
            & (width_optima["domain"] == domain)
        ].copy()
        xs = np.geomspace(min(POW2), max(POW2), 256)
        for fit_name, label in fit_order:
            fit_row = fit_table[
                (fit_table["model_short"] == model)
                & (fit_table["domain"] == domain)
                & (fit_table["fit_name"] == fit_name)
            ].iloc[0]
            ys = width_fit_predict(fit_row, xs)
            ax.plot(xs, ys, linewidth=2.0, color=FIT_COLORS[fit_name], label=label)
        ax.scatter(opt["compute"], opt["width"], color="black", marker="*", s=150, zorder=5, label="Exact optimum")
        ax.set_xscale("log", base=2)
        ax.set_xticks(POW2)
        ax.set_yscale("log", base=2)
        ax.set_yticks(WIDTH_TICKS)
        
        global_max_w = width_optima["width"].max() if not width_optima.empty else 128
        ax.set_ylim(bottom=2**-0.5, top=global_max_w * (2**0.5))
        
        ax.set_title(f"{MODEL_LABELS[model]} | {DOMAIN_LABELS[domain]}")
        ax.set_xlabel(r"Budget $N$")
        ax.set_ylabel(r"Optimal width $w^*$")
    handles, labels = collect_unique_legend_items(list(axes))
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_figure(fig, "fig_width_fit_compare")
    plt.close(fig)


def plot_bk_gap_strip(
    bk_gap_table: pd.DataFrame,
    model_short: str,
    domain: str,
    basename: str,
) -> None:
    set_plot_style()
    budgets = [128, 64, 32]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)

    for ax, fixed_budget in zip(axes, budgets):
        sub = bk_gap_table[
            (bk_gap_table["model_short"] == model_short)
            & (bk_gap_table["domain"] == domain)
            & (bk_gap_table["compute"] == fixed_budget)
            & (bk_gap_table["algorithm"] == "pbeam")
        ].copy()
        if sub.empty:
            ax.set_title(f"$N={fixed_budget}$ (no data)")
            continue

        width_order = sorted(sub["width"].unique())
        sub["width_cat"] = pd.Categorical(sub["width"], categories=width_order, ordered=True)

        sns.stripplot(
            data=sub,
            x="width_cat",
            y="train_acc01_mean",
            hue="width",
            palette=PALETTE,
            jitter=0.12,
            alpha=0.55,
            size=9,
            linewidth=0.5,
            edgecolor="white",
            legend=False,
            ax=ax,
            zorder=2,
        )

        best_per_width = sub.groupby("width", as_index=False)["train_acc01_mean"].max()
        best_per_width = best_per_width.sort_values("width")
        width_to_x = {w: i for i, w in enumerate(width_order)}
        ridge_x = [width_to_x[w] for w in best_per_width["width"]]
        ridge_y = best_per_width["train_acc01_mean"].tolist()

        ax.plot(ridge_x, ridge_y, color="black", linewidth=2.5, zorder=3)
        ax.scatter(ridge_x, ridge_y, color="black", marker="*", s=180, zorder=4)

        ax.set_title(r"$\mathbf{N=" + str(fixed_budget) + r"}$", fontsize=13)
        ax.set_xlabel(r"Total Width $w = n \times k$")
        if ax == axes[0]:
            ax.set_ylabel(r"Train Accuracy ($\mathrm{Acc}_{0.1}$)")
        else:
            ax.set_ylabel("")

    fig.suptitle(
        f"{MODEL_LABELS.get(model_short, model_short)} | "
        f"{DOMAIN_LABELS.get(domain, domain)}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, basename)
    plt.close(fig)


def plot_algorithm_delta(alg_delta: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True, sharey=True)
    panel_order = [
        ("gpt-oss-20b", "bio"),
        ("gpt-oss-20b", "material"),
        ("qwen3-30b-a3b", "bio"),
    ]
    flat_axes = np.atleast_1d(axes)
    for ax, (model, domain) in zip(flat_axes, panel_order):
        sub = alg_delta[
            (alg_delta["model_short"] == model)
            & (alg_delta["domain"] == domain)
        ].copy()
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        for width in sorted(sub["width"].unique()):
            width_df = sub[sub["width"] == width].sort_values("compute")
            ax.plot(
                width_df["compute"],
                width_df["pie_minus_pbeam_pp"],
                marker="o",
                linewidth=1.8,
                markersize=4.2,
                color=PALETTE.get(int(width), "#999999"),
                alpha=0.92,
                label=f"$w={int(width)}$",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(POW2)
        ax.set_title(f"{MODEL_LABELS[model]} | {DOMAIN_LABELS[domain]}")
        ax.set_xlabel(r"Budget $N$")
        ax.set_ylabel(r"PIE $-$ PBeam (train, pp)")
    handles, labels = collect_unique_legend_items([flat_axes[0]])
    fig.legend(handles, labels, ncol=5, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_figure(fig, "fig_algorithm_delta")
    plt.close(fig)


def plot_openevolve_compare(openevolve_compare: pd.DataFrame) -> None:
    if openevolve_compare.empty:
        return
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.6), sharex=True, sharey=False)
    panel_order = [
        ("bio", 16),
        ("bio", 32),
        ("material", 16),
        ("material", 32),
    ]
    algorithm_colors = {
        "pbeam": "#1f77b4",
        "pie": "#ff7f0e",
        "openevolve": "#d62728",
    }
    algorithm_markers = {
        "pbeam": "o",
        "pie": "s",
        "openevolve": "^",
    }
    for ax, (domain, width) in zip(axes.flatten(), panel_order):
        sub = openevolve_compare[
            (openevolve_compare["domain"] == domain)
            & (openevolve_compare["width"] == width)
        ].copy()
        if sub.empty:
            ax.axis("off")
            continue
        for algorithm in ["pbeam", "pie", "openevolve"]:
            alg_sub = sub[sub["algorithm"] == algorithm].sort_values("compute")
            if alg_sub.empty:
                continue
            ax.plot(
                alg_sub["compute"],
                alg_sub["train_acc01_mean"],
                marker=algorithm_markers[algorithm],
                linewidth=1.9,
                markersize=4.5,
                color=algorithm_colors[algorithm],
                alpha=0.95,
                label=ALGORITHM_LABELS[algorithm],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 32, 64, 128])
        ax.set_title(f"{DOMAIN_LABELS[domain]} | $w={width}$")
        ax.set_xlabel(r"Budget $N$")
        ax.set_ylabel(r"Mean train $\mathrm{Acc}_{0.1}$")
    handles, labels = collect_unique_legend_items(list(axes.flatten()))
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig_openevolve_compare")
    plt.close(fig)


def plot_group_tradeoff(group_tradeoff_plot: pd.DataFrame) -> None:
    if group_tradeoff_plot.empty:
        return
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4), sharex=False, sharey=False)
    panel_order = [
        ("bio", "train_acc_delta_pp", r"Train $\Delta \mathrm{Acc}_{0.1}$ (pp)"),
        ("bio", "speedup_vs_G1", r"Speedup vs. $g=1$"),
        ("material", "train_acc_delta_pp", r"Train $\Delta \mathrm{Acc}_{0.1}$ (pp)"),
        ("material", "speedup_vs_G1", r"Speedup vs. $g=1$"),
    ]
    for ax, (domain, value_col, ylabel) in zip(axes.flatten(), panel_order):
        sub = group_tradeoff_plot[group_tradeoff_plot["domain"] == domain]
        for width in sorted(sub["width"].unique()):
            width_df = sub[sub["width"] == width].sort_values("G")
            ax.plot(
                width_df["G"],
                width_df[value_col],
                marker="o",
                linewidth=1.9,
                markersize=4.5,
                color=PALETTE.get(int(width), "#999999"),
                label=f"$w={int(width)}$",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub["G"].unique()))
        ax.set_title(f"{DOMAIN_LABELS[domain]} | $N={HIGH_BUDGET}$")
        ax.set_xlabel(r"Group size $g$")
        ax.set_ylabel(ylabel)
        if value_col == "train_acc_delta_pp":
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        else:
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    handles, labels = collect_unique_legend_items(list(axes.flatten()))
    fig.legend(handles, labels, ncol=5, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig_group_tradeoff")
    plt.close(fig)


def plot_time_frontier(
    frontier_df: pd.DataFrame,
    best_with_group: pd.DataFrame,
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharex=False, sharey=False)
    markers = {"pbeam": "o", "pie": "s"}
    panel_specs = [
        ("gpt-oss-20b", "bio", frontier_df[frontier_df["domain"] == "bio"].copy(), best_with_group[best_with_group["domain"] == "bio"].copy(), True),
        ("gpt-oss-20b", "material", frontier_df[frontier_df["domain"] == "material"].copy(), best_with_group[best_with_group["domain"] == "material"].copy(), True),
    ]
    highlight_budgets = {1, 8, 128}
    for ax, (model, domain, sub, labels_df, show_group) in zip(axes, panel_specs):
        pareto = add_frontier_flag(sub, "wall_time_s_mean", "train_acc01_mean")
        
        for algorithm in ["pbeam", "pie"]:
            alg_sub = pareto[pareto["algorithm"] == algorithm]
            
            non_front = alg_sub[~alg_sub["is_pareto"]]
            if not non_front.empty:
                ax.scatter(
                    non_front["wall_time_s_mean"],
                    non_front["train_acc01_mean"],
                    c=[PALETTE.get(int(w), "#999999") for w in non_front["width"]],
                    s=58 * 0.7,
                    alpha=0.25,
                    marker=markers[algorithm],
                    edgecolors="none",
                    linewidth=0,
                    label=ALGORITHM_LABELS[algorithm],
                    zorder=2,
                )
                
            front = alg_sub[alg_sub["is_pareto"]]
            if not front.empty:
                ax.scatter(
                    front["wall_time_s_mean"],
                    front["train_acc01_mean"],
                    c=[PALETTE.get(int(w), "#999999") for w in front["width"]],
                    s=58 * 1.5,
                    alpha=1.0,
                    marker=markers[algorithm],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=11,
                    label=ALGORITHM_LABELS[algorithm],
                )
                
        pareto_front = pareto[pareto["is_pareto"]].sort_values("wall_time_s_mean")
        ax.plot(
            pareto_front["wall_time_s_mean"],
            pareto_front["train_acc01_mean"],
            color="black",
            linewidth=2.5,
            zorder=10,
        )
        
        global_pareto = pareto[pareto["is_pareto"]].copy()
        annotated_rows = []
        # N=1 from Pareto
        if not global_pareto.empty:
            idx = (np.log2(global_pareto["compute"]) - np.log2(1)).abs().idxmin()
            annotated_rows.append(global_pareto.loc[idx])
            
        # User-specified points for N=128, w=32
        if domain == "bio":
            # Bio: N=128, w=32, G=2 (PIE, Acc=0.9903, Time=1635.9)
            pt1 = sub[(sub["compute"] == 128) & (sub["width"] == 32) & (sub["G"] == 2) & (sub["algorithm"] == "pie")]
            if not pt1.empty: annotated_rows.append(pt1.iloc[0])
            # Bio: N=128, w=32, G=4 (PIE, Acc=0.9885, Time=1622.1)
            pt2 = sub[(sub["compute"] == 128) & (sub["width"] == 32) & (sub["G"] == 4) & (sub["algorithm"] == "pie")]
            if not pt2.empty: annotated_rows.append(pt2.iloc[0])
        else:
            # Material: N=128, w=32, G=2 (PIE, Acc=0.9989, Time=2687.2)
            pt1 = sub[(sub["compute"] == 128) & (sub["width"] == 32) & (sub["G"] == 2) & (sub["algorithm"] == "pie")]
            if not pt1.empty: annotated_rows.append(pt1.iloc[0])
            # Material: N=128, w=32, G=4 (PBeam, Acc=0.9975, Time=2649.6)
            pt2 = sub[(sub["compute"] == 128) & (sub["width"] == 32) & (sub["G"] == 4) & (sub["algorithm"] == "pbeam")]
            if not pt2.empty: annotated_rows.append(pt2.iloc[0])

        labels_df = pd.DataFrame(annotated_rows).drop_duplicates(subset=["wall_time_s_mean", "algorithm"])
        
        print("\n=== ANNOTATED POINTS ===")
        print(f"Model: {model}, Domain: {domain}")
        print(labels_df[["algorithm", "compute", "width", "G", "B", "K", "train_acc01_mean", "wall_time_s_mean"]].to_string())
        
        for _, row in labels_df.sort_values("compute").iterrows():
            c_val = int(row["compute"])
            w_val = int(row["width"])
            g_val = int(row["G"])
            b_val = int(row["B"])
            k_val = int(row["K"])
            alg_label = "PBeam(PIE)" if (c_val == 1 or w_val == c_val) else ALGORITHM_LABELS[row["algorithm"]]
            label = f"N={c_val}, {alg_label}, w={w_val}"
            if show_group:
                label += f", g={g_val}"
                
            # Add time and accuracy for points with budget > 1
            if c_val != 1:
                label += f"\nTime={row['wall_time_s_mean']:.0f}s, $\\mathrm{{Acc}}_{{0.1}}$={row['train_acc01_mean']:.3f}"
            
            if c_val == 128:
                if g_val == 4:
                    xoff, yoff = (-10, -110)
                    ha = "right"
                else:
                    xoff, yoff = (30, -75)
                    ha = "left"
            elif c_val == 1:
                xoff, yoff = (20, 20)
                ha = "left"
            elif g_val > 1:
                xoff, yoff = (-5, -90)
                ha = "right"
            else:
                xoff, yoff = (25, -55)
                ha = "left"
                
            ax.annotate(
                label,
                (row["wall_time_s_mean"], row["train_acc01_mean"]),
                fontsize=10.3,
                xytext=(xoff, yoff),
                textcoords="offset points",
                ha=ha,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
                arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.2, shrinkA=3, shrinkB=6, alpha=0.8),
                zorder=14,
            )
            
        ax.set_ylim(sub["train_acc01_mean"].min() - 0.015, min(1.005, sub["train_acc01_mean"].max() + 0.012))
        ax.set_xscale("log")
        ax.set_title(f"{MODEL_LABELS[model]} | {DOMAIN_LABELS[domain]}")
        ax.set_xlabel("Mean wall-clock time (s)")
        ax.set_ylabel(r"Mean train $\mathrm{Acc}_{0.1}$")
    handles, labels = collect_unique_legend_items(list(np.atleast_1d(axes)))
    for w in [1, 2, 4, 8, 16, 32, 64, 128]:
        dummy = axes[0].scatter([], [], color=PALETTE[w], s=60, marker="o", edgecolors="none")
        handles.append(dummy)
        labels.append(f"$w={w}$")
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save_figure(fig, "fig_time_frontier")
    plt.close(fig)


def plot_variance_gap(std_df: pd.DataFrame) -> None:
    set_plot_style()
    plot_df = std_df.copy()
    configs = [
        ("gpt-oss-20b", "bio"),
        ("gpt-oss-20b", "material"),
    ]
    fig, axes = plt.subplots(1, len(configs), figsize=(12.5, 5.2), sharex=False, sharey=False)
    
    for i, (model, domain) in enumerate(configs):
        ax = axes[i]
        sub = plot_df[
            (plot_df["model_short"].str.lower() == model.lower()) & 
            (plot_df["domain"].str.lower() == domain.lower())
        ].sort_values("compute")
        
        if sub.empty:
            continue
            
        ax.plot(
            sub["compute"],
            sub["train_acc01_mean"],
            marker="o",
            label="Train Accuracy",
            color="#2980B9",
            linewidth=2.2,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            zorder=4
        )
        ax.plot(
            sub["compute"],
            sub["test_acc01_mean"],
            marker="s",
            label="Test Accuracy",
            color="#E67E22",
            linewidth=2.2,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            zorder=3
        )
        
        ax.set_xscale("log", base=2)
        ax.set_xticks(POW2)
        ax.set_title(f"{MODEL_LABELS.get(model, model)} | {DOMAIN_LABELS.get(domain, domain)}")
        ax.set_xlabel(r"Budget $N$")
        if i == 0:
            ax.set_ylabel(r"Mean Accuracy ($\mathrm{Acc}_{0.1}$)")
        ax.grid(True, alpha=0.3, linestyle="--")
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save_figure(fig, "fig_variance_gap")
    plt.close(fig)


def plot_cross_model(width_optima: pd.DataFrame, width_optima_tol: pd.DataFrame) -> None:
    set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.5), sharex=True, sharey=True)
    domain = "bio"
    for model in ["gpt-oss-20b", "qwen3-30b-a3b"]:
        opt = width_optima[
            (width_optima["model_short"] == model)
            & (width_optima["domain"] == domain)
        ].copy()
        opt_tol = width_optima_tol[
            (width_optima_tol["model_short"] == model)
            & (width_optima_tol["domain"] == domain)
        ].copy()
        ax.plot(
            opt["compute"],
            opt["width"],
            marker="o",
            linewidth=1.9,
            color=MODEL_COLORS[model],
            label=f"{MODEL_LABELS[model]} exact",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(POW2)
    ax.set_yscale("log", base=2)
    ax.set_yticks(WIDTH_TICKS)
    ax.set_title("Bio")
    ax.set_xlabel(r"Budget $N$")
    ax.set_ylabel(r"Optimal width $w^*$")
    handles, labels = collect_unique_legend_items([ax])
    fig.legend(handles, labels, ncol=2, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, "fig_cross_model")
    plt.close(fig)


def width_fit_to_row(model: str, domain: str, result: FitResult) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "model_short": model,
        "domain": domain,
        "fit_name": result.name,
        "r2": result.r2,
        "mae": result.mae,
        "rmse": result.rmse,
        "fit_min_budget": result.fit_min_budget,
        "param_a": np.nan,
        "param_alpha": np.nan,
        "param_bias": np.nan,
        "formula": "",
    }
    if result.name == "log_power_budget":
        row["param_a"] = result.params["c"]
        row["param_alpha"] = result.params["alpha"]
        row["formula"] = f"w*= {result.params['c']:.6g} * (log2 N)^{result.params['alpha']:.4f}"
    elif result.name == "hinge_log_power_budget":
        row["param_a"] = result.params["c"]
        row["param_alpha"] = result.params["alpha"]
        row["formula"] = f"w*= 1 + {result.params['c']:.6g} * [log2 N - 1]_+^{result.params['alpha']:.4f}"
    elif result.name == "anchored_log_power_budget":
        row["param_a"] = result.params["c"]
        row["param_alpha"] = result.params["alpha"]
        row["formula"] = f"w*= 1 + {result.params['c']:.6g} * (log2 N)^{result.params['alpha']:.4f}"
    elif result.name == "exp_log_power_budget":
        row["param_a"] = result.params["c"]
        row["param_alpha"] = result.params["alpha"]
        row["formula"] = f"w*= exp({result.params['c']:.6g} * (log2 N)^{result.params['alpha']:.4f})"
    elif result.name == "hinge_exp_log_power_budget":
        row["param_a"] = result.params["c"]
        row["param_alpha"] = result.params["alpha"]
        row["formula"] = f"w*= exp({result.params['c']:.6g} * [log2 N - 1]_+^{result.params['alpha']:.4f})"
    elif result.name == "cubic_logwidth_budget":
        row["param_a"] = result.params["a1"]
        row["param_alpha"] = result.params["a2"]
        row["param_bias"] = result.params["a3"]
        row["formula"] = (
            f"w*= exp({result.params['a1']:.6g} * L + "
            f"{result.params['a2']:.6g} * L^2 + {result.params['a3']:.6g} * L^3), L=log2 N"
        )
    elif result.name == "hinge_cubic_logwidth_budget":
        row["param_a"] = result.params["a1"]
        row["param_alpha"] = result.params["a2"]
        row["param_bias"] = result.params["a3"]
        row["formula"] = (
            f"w*= exp({result.params['a1']:.6g} * s + "
            f"{result.params['a2']:.6g} * s^2 + {result.params['a3']:.6g} * s^3), s=[log2 N - 1]_+"
        )
    elif result.name == "shifted_log_power_budget":
        row["param_a"] = result.params["c"]
        row["param_alpha"] = result.params["alpha"]
        row["param_bias"] = result.params["bias"]
        row["formula"] = (
            f"w*= {result.params['bias']:.6g} + "
            f"{result.params['c']:.6g} * (log2 N)^{result.params['alpha']:.4f}"
        )
    elif result.name == "power_law":
        row["param_a"] = result.params["c"]
        row["param_alpha"] = result.params["alpha"]
        row["formula"] = f"w*= {result.params['c']:.6g} * N^{result.params['alpha']:.4f}"
    elif result.name == "sqrt_law":
        row["param_a"] = result.params["c"]
        row["param_alpha"] = 0.5
        row["formula"] = f"w*= {result.params['c']:.6g} * sqrt(N)"
    return row


def perf_fit_to_row(model: str, domain: str, result: FitResult) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "model_short": model,
        "domain": domain,
        "fit_name": result.name,
        "r2": result.r2,
        "mae": result.mae,
        "rmse": result.rmse,
        "param_a": np.nan,
        "param_alpha": np.nan,
        "formula": "",
    }
    if result.name == "gap_power_law":
        row["param_a"] = result.params["c"]
        row["param_alpha"] = result.params["alpha"]
        row["formula"] = f"1 - Acc = {result.params['c']:.6g} * N^(-{result.params['alpha']:.4f})"
    elif result.name == "log_linear_acc":
        row["param_a"] = result.params["a"]
        row["param_alpha"] = result.params["b"]
        row["formula"] = f"Acc = {result.params['a']:.6g} + {result.params['b']:.6g} * log2 N"
    return row


def main() -> None:
    ensure_dirs()
    set_plot_style()
    
    MAIN_MODEL = "gpt-oss-20b"
    main_df = fill_theoretical_pie_equivalents(load_main_results())
    group_df = load_group_results()

    main_pow2 = main_df[main_df["budget_pow2"]].copy()
    group_pow2 = group_df[group_df["budget_pow2"]].copy()

    main_seed = domain_seed_average(main_pow2)
    group_seed = domain_seed_average(group_pow2)
    main_summary = summarize_over_seeds(main_seed)
    group_summary = summarize_over_seeds(group_seed)

    main_summary.to_csv(TABLE_DIR / "main_summary.csv", index=False)
    group_summary.to_csv(TABLE_DIR / "group_summary.csv", index=False)

    # Width study: first optimize n/k within each algorithm, then optimize over PBeam and PIE.
    best_bk_per_alg = choose_best_config(
        main_summary,
        ["model_short", "domain", "algorithm", "width", "compute"],
    )
    width_frontier = choose_best_config(
        best_bk_per_alg,
        ["model_short", "domain", "width", "compute"],
    )
    width_optima = choose_best_config(
        width_frontier,
        ["model_short", "domain", "compute"],
    )
    width_optima_tol = choose_smallest_within_tolerance(
        width_frontier,
        ["model_short", "domain", "compute"],
        tolerance_abs=0.001,
    )
    width_frontier.to_csv(TABLE_DIR / "width_frontier.csv", index=False)
    width_optima.to_csv(TABLE_DIR / "width_optima.csv", index=False)
    width_optima_tol.to_csv(TABLE_DIR / "width_optima_tol_0p1pp.csv", index=False)

    width_fit_rows: list[dict[str, float | str]] = []
    for (model, domain), sub in width_optima.groupby(["model_short", "domain"]):
        x = sub["compute"].to_numpy(dtype=float)
        y = sub["width"].to_numpy(dtype=float)
        fitters = [
            fit_log_power_width_law,
            fit_hinge_log_power_width_law,
            fit_anchored_log_power_width_law,
            fit_exp_log_power_width_law,
            fit_hinge_exp_log_power_width_law,
            fit_cubic_logwidth_law,
            fit_hinge_cubic_logwidth_law,
            fit_shifted_log_power_width_law,
            fit_power_law,
            fit_sqrt_law,
        ]
        for fitter in fitters:
            result = fitter(x, y)
            width_fit_rows.append(width_fit_to_row(model, domain, result))
    width_fit_table = pd.DataFrame(width_fit_rows).sort_values(
        ["model_short", "domain", "mae", "r2"],
        ascending=[True, True, True, False],
    )
    width_fit_table.to_csv(TABLE_DIR / "width_scaling_fits.csv", index=False)

    # Performance scaling law under the best overall allocation.
    best_overall = choose_best_config(
        main_summary,
        ["model_short", "domain", "compute"],
    )
    best_overall.to_csv(TABLE_DIR / "best_overall_allocation.csv", index=False)
    perf_fit_rows: list[dict[str, float | str]] = []
    for (model, domain), sub in best_overall.groupby(["model_short", "domain"]):
        x = sub["compute"].to_numpy(dtype=float)
        y = sub["train_acc01_mean"].to_numpy(dtype=float)
        for fitter in (fit_gap_power_law, fit_log_linear_acc):
            result = fitter(x, y)
            perf_fit_rows.append(perf_fit_to_row(model, domain, result))
    perf_fit_table = pd.DataFrame(perf_fit_rows).sort_values(
        ["model_short", "domain", "r2"],
        ascending=[True, True, False],
    )
    perf_fit_table.to_csv(TABLE_DIR / "performance_scaling_fits.csv", index=False)

    # B/K analysis: gap to the best slice after fixing model, domain, algorithm, width, and budget.
    bk = main_summary.copy()
    slice_best = bk.groupby(["model_short", "domain", "algorithm", "width", "compute"])["train_acc01_mean"].transform("max")
    bk["gap_to_best_pp"] = 100.0 * (slice_best - bk["train_acc01_mean"])
    bk.to_csv(TABLE_DIR / "bk_gap_table.csv", index=False)

    bk_reco = (
        bk.groupby(["model_short", "domain", "algorithm", "width", "B", "K", "beam_fraction"], as_index=False)
        .agg(
            mean_gap_pp=("gap_to_best_pp", "mean"),
            max_gap_pp=("gap_to_best_pp", "max"),
            mean_train_acc01=("train_acc01_mean", "mean"),
        )
        .sort_values(
            ["model_short", "domain", "algorithm", "width", "mean_gap_pp", "B", "K"],
            ascending=[True, True, True, True, True, True, True],
        )
        .groupby(["model_short", "domain", "algorithm", "width"], as_index=False)
        .first()
    )
    bk_reco.to_csv(TABLE_DIR / "bk_recommendation.csv", index=False)

    bk_winners = choose_best_config(
        main_summary,
        ["model_short", "domain", "algorithm", "width", "compute"],
    ).copy()

    def bk_category(beam_fraction: float) -> str:
        if beam_fraction < 0.125:
            return "sample-heavy"
        if beam_fraction <= 0.5:
            return "balanced"
        return "beam-heavy"

    bk_winners["allocation_class"] = bk_winners["beam_fraction"].map(bk_category)
    bk_category_summary = (
        bk_winners.groupby(["model_short", "domain", "allocation_class"], as_index=False)
        .size()
        .rename(columns={"size": "winner_count"})
        .sort_values(["model_short", "domain", "winner_count"], ascending=[True, True, False])
    )
    bk_category_summary.to_csv(TABLE_DIR / "bk_category_summary.csv", index=False)

    # Algorithm comparison after optimizing n/k within each algorithm.
    alg_wide = (
        best_bk_per_alg.pivot_table(
            index=["model_short", "domain", "width", "compute"],
            columns="algorithm",
            values=["train_acc01_mean", "test_acc01_mean"],
            aggfunc="first",
        )
        .reset_index()
    )
    alg_wide.columns = [
        "_".join([str(c) for c in col if str(c) != ""]).strip("_")
        if isinstance(col, tuple)
        else col
        for col in alg_wide.columns
    ]
    alg_wide["pie_minus_pbeam_pp"] = 100.0 * (
        alg_wide["train_acc01_mean_pie"] - alg_wide["train_acc01_mean_pbeam"]
    )
    alg_wide["pie_minus_pbeam_test_pp"] = 100.0 * (
        alg_wide["test_acc01_mean_pie"] - alg_wide["test_acc01_mean_pbeam"]
    )
    alg_wide.to_csv(TABLE_DIR / "algorithm_delta.csv", index=False)

    alg_global = (
        choose_best_config(main_summary, ["model_short", "domain", "algorithm", "compute"])
        .pivot_table(
            index=["model_short", "domain", "compute"],
            columns="algorithm",
            values=["train_acc01_mean", "test_acc01_mean"],
            aggfunc="first",
        )
        .reset_index()
    )
    alg_global.columns = [
        "_".join([str(c) for c in col if str(c) != ""]).strip("_")
        if isinstance(col, tuple)
        else col
        for col in alg_global.columns
    ]
    alg_global["pie_minus_pbeam_pp"] = 100.0 * (
        alg_global["train_acc01_mean_pie"] - alg_global["train_acc01_mean_pbeam"]
    )
    alg_global["pie_minus_pbeam_test_pp"] = 100.0 * (
        alg_global["test_acc01_mean_pie"] - alg_global["test_acc01_mean_pbeam"]
    )
    alg_global.to_csv(TABLE_DIR / "algorithm_delta_global.csv", index=False)

    alg_delta_summary = (
        alg_wide.groupby(["model_short", "domain"], as_index=False)
        .agg(
            mean_abs_train_delta_pp=("pie_minus_pbeam_pp", lambda x: float(np.mean(np.abs(x)))),
            mean_signed_train_delta_pp=("pie_minus_pbeam_pp", "mean"),
            mean_abs_test_delta_pp=("pie_minus_pbeam_test_pp", lambda x: float(np.mean(np.abs(x)))),
        )
        .sort_values(["model_short", "domain"])
    )
    alg_delta_summary.to_csv(TABLE_DIR / "algorithm_delta_summary.csv", index=False)

    openevolve_style = group_summary[
        (group_summary["model_short"] == MAIN_MODEL)
        & (group_summary["algorithm"] == "pbeam")
        & (group_summary["K"] == 1)
        & (group_summary["G"] == group_summary["B"])
    ].copy()
    openevolve_style["algorithm"] = "openevolve"
    openevolve_style = openevolve_style[
        [
            "model_short",
            "domain",
            "algorithm",
            "width",
            "compute",
            "iteration",
            "B",
            "K",
            "G",
            "train_acc01_mean",
            "train_acc01_std",
            "test_acc01_mean",
            "test_acc01_std",
            "wall_time_s_mean",
            "wall_time_s_std",
        ]
    ].copy()

    openevolve_compare = pd.concat(
        [
            best_bk_per_alg[best_bk_per_alg["model_short"] == MAIN_MODEL].copy(),
            openevolve_style,
        ],
        ignore_index=True,
    )
    if not openevolve_style.empty:
        valid_widths = sorted(openevolve_style["width"].unique())
        openevolve_compare = openevolve_compare[openevolve_compare["width"].isin(valid_widths)].copy()
    openevolve_compare.to_csv(TABLE_DIR / "openevolve_compare.csv", index=False)

    if not openevolve_compare.empty:
        slice_best = openevolve_compare.groupby(["domain", "width", "compute"])["train_acc01_mean"].transform("max")
        slice_fastest = openevolve_compare.groupby(["domain", "width", "compute"])["wall_time_s_mean"].transform("min")
        openevolve_compare["gap_to_slice_best_pp"] = 100.0 * (
            slice_best - openevolve_compare["train_acc01_mean"]
        )
        budget_best = openevolve_compare.groupby(["domain", "compute"])["train_acc01_mean"].transform("max")
        openevolve_compare["gap_to_budget_best_pp"] = 100.0 * (
            budget_best - openevolve_compare["train_acc01_mean"]
        )
        openevolve_compare["slowdown_vs_fastest"] = (
            openevolve_compare["wall_time_s_mean"] / slice_fastest
        )
    openevolve_compare.to_csv(TABLE_DIR / "openevolve_compare.csv", index=False)

    openevolve_summary = (
        openevolve_compare.groupby(["domain", "algorithm"], as_index=False)
        .agg(
            mean_gap_to_slice_best_pp=("gap_to_slice_best_pp", "mean"),
            mean_gap_to_budget_best_pp=("gap_to_budget_best_pp", "mean"),
            max_gap_to_slice_best_pp=("gap_to_slice_best_pp", "max"),
            mean_slowdown_vs_fastest=("slowdown_vs_fastest", "mean"),
            mean_train_acc01=("train_acc01_mean", "mean"),
            mean_test_acc01=("test_acc01_mean", "mean"),
            count=("compute", "count"),
        )
        .sort_values(["domain", "algorithm"])
    )
    openevolve_summary.to_csv(TABLE_DIR / "openevolve_compare_summary.csv", index=False)

    # Group analysis on gpt-oss only. Compare against the g=1 run with the same (algorithm, n, k, N).
    group_focus = group_summary[group_summary["model_short"] == MAIN_MODEL].copy()
    baseline = (
        group_focus[group_focus["G"] == 1][
            ["domain", "algorithm", "B", "K", "compute", "train_acc01_mean", "wall_time_s_mean"]
        ]
        .rename(
            columns={
                "train_acc01_mean": "train_acc01_g1",
                "wall_time_s_mean": "wall_time_s_g1",
            }
        )
    )
    group_tradeoff = group_focus.merge(
        baseline,
        on=["domain", "algorithm", "B", "K", "compute"],
        how="left",
    )
    group_tradeoff["train_acc_delta_pp"] = 100.0 * (
        group_tradeoff["train_acc01_mean"] - group_tradeoff["train_acc01_g1"]
    )
    group_tradeoff["speedup_vs_G1"] = group_tradeoff["wall_time_s_g1"] / group_tradeoff["wall_time_s_mean"]
    group_tradeoff.to_csv(TABLE_DIR / "group_tradeoff.csv", index=False)

    group_tradeoff_n128 = group_tradeoff[group_tradeoff["compute"] == HIGH_BUDGET].copy()
    group_tradeoff_n128.to_csv(TABLE_DIR / "group_tradeoff_n128.csv", index=False)

    group_plot = choose_best_config(
        group_focus[group_focus["compute"] == HIGH_BUDGET],
        ["domain", "width", "G"],
    )
    group_plot_baseline = (
        group_plot[group_plot["G"] == 1][["domain", "width", "train_acc01_mean", "wall_time_s_mean"]]
        .rename(
            columns={
                "train_acc01_mean": "train_acc01_g1_width",
                "wall_time_s_mean": "wall_time_s_g1_width",
            }
        )
    )
    group_plot = group_plot.merge(group_plot_baseline, on=["domain", "width"], how="left")
    group_plot["train_acc_delta_pp"] = 100.0 * (
        group_plot["train_acc01_mean"] - group_plot["train_acc01_g1_width"]
    )
    group_plot["speedup_vs_G1"] = group_plot["wall_time_s_g1_width"] / group_plot["wall_time_s_mean"]
    group_plot.to_csv(TABLE_DIR / "group_tradeoff_width_best_n128.csv", index=False)

    group_safe_speedups = (
        group_tradeoff_n128[group_tradeoff_n128["G"] > 1]
        .sort_values(
            ["domain", "width", "speedup_vs_G1", "train_acc_delta_pp"],
            ascending=[True, True, False, False],
        )
        .groupby(["domain", "width"], as_index=False)
        .first()
    )
    group_safe_speedups.to_csv(TABLE_DIR / "group_safe_speedups.csv", index=False)

    # Time frontier with group variants included for gpt-oss.
    extra_groups = group_summary[group_summary["G"] > 1].copy()
    time_candidates = pd.concat(
        [
            main_summary[main_summary["model_short"] == MAIN_MODEL],
            extra_groups,
        ],
        ignore_index=True,
    )
    time_best = choose_best_config(
        time_candidates,
        ["model_short", "domain", "compute", "algorithm", "width", "G"],
    )
    time_best = time_best[time_best["model_short"] == MAIN_MODEL].copy()
    time_best.to_csv(TABLE_DIR / "time_candidates.csv", index=False)

    best_with_group = choose_best_config(
        time_candidates[time_candidates["model_short"] == MAIN_MODEL],
        ["model_short", "domain", "compute"],
    )
    best_with_group.to_csv(TABLE_DIR / "best_overall_with_group_gptoss.csv", index=False)

    # Seed std and generalization gap of the best overall allocation.
    best_std = best_overall[
        [
            "model_short",
            "domain",
            "compute",
            "train_acc01_mean",
            "train_acc01_std",
            "test_acc01_mean",
            "test_acc01_std",
            "B",
            "K",
            "algorithm",
            "width",
        ]
    ].copy()
    best_std.to_csv(TABLE_DIR / "best_allocation_seed_std.csv", index=False)

    # Figures for the paper and appendix.
    plot_width_frontier_main(width_frontier, width_optima)
    plot_width_scaling_main(width_frontier, width_optima, width_optima_tol, width_fit_table)
    plot_width_frontier_all(width_frontier, width_optima)
    # BK Gap analysis for PBeam
    bk = main_summary[main_summary["algorithm"] == "pbeam"].copy()
    plot_bk_gap_strip(bk, MAIN_MODEL, "bio", "fig_bk_gap_bio")
    plot_bk_gap_strip(bk, MAIN_MODEL, "material", "fig_bk_gap_material")

    # Allocation ranking table as requested
    rank_cols = ["model_short", "domain", "compute", "width", "algorithm", "B", "K", "train_acc01_mean", "test_acc01_mean"]
    ranking = main_summary[
        (main_summary["model_short"] == MAIN_MODEL) & 
        (main_summary["algorithm"].isin(["pbeam", "pie"]))
    ][rank_cols].copy()
    ranking = ranking.sort_values(["domain", "compute", "width", "algorithm", "train_acc01_mean"], ascending=[True, True, True, True, False])
    ranking.to_csv(TABLE_DIR / "allocation_ranking.csv", index=False)
    plot_algorithm_delta(alg_wide)
    plot_openevolve_compare(openevolve_compare)
    plot_group_tradeoff(group_plot)
    plot_time_frontier(time_best, best_with_group)
    plot_variance_gap(best_std)
    plot_cross_model(width_optima, width_optima_tol)

    # Compact draft summary.
    lines: list[str] = []
    lines.append("# TTS Compute Allocation Analysis Summary")
    lines.append("")
    lines.append("## Width law candidates (exact optimum, best over PBeam/PIE)")
    for (model, domain), sub in width_fit_table.groupby(["model_short", "domain"]):
        lines.append(f"- {model} | {domain}")
        for _, row in sub.sort_values(["mae", "r2"], ascending=[True, False]).head(4).iterrows():
            lines.append(
                f"  {row['fit_name']}: R^2={row['r2']:.3f}, MAE={row['mae']:.3f}, "
                f"RMSE={row['rmse']:.3f}, {row['formula']}"
            )
    lines.append("")
    lines.append("## Best overall allocation by budget")
    for (model, domain), sub in best_overall.groupby(["model_short", "domain"]):
        lines.append(f"- {model} | {domain}")
        for _, row in sub.sort_values("compute").iterrows():
            lines.append(
                f"  N={int(row['compute'])}: {ALGORITHM_LABELS[row['algorithm']]}, "
                f"n={int(row['B'])}, k={int(row['K'])}, T={int(row['iteration'])}, "
                f"w={int(row['width'])}, train={row['train_acc01_mean']:.4f}±{row['train_acc01_std']:.4f}, "
                f"test={row['test_acc01_mean']:.4f}±{row['test_acc01_std']:.4f}"
            )
    lines.append("")
    lines.append("## PIE vs PBeam")
    for _, row in alg_delta_summary.iterrows():
        lines.append(
            f"- {row['model_short']} | {row['domain']}: "
            f"mean |train delta| = {row['mean_abs_train_delta_pp']:.3f} pp, "
            f"mean signed delta = {row['mean_signed_train_delta_pp']:.3f} pp"
        )
    lines.append("")
    lines.append("## OpenEvolve matched comparison")
    for _, row in openevolve_summary.iterrows():
        lines.append(
            f"- {row['domain']} | {ALGORITHM_LABELS[row['algorithm']]}: "
            f"mean gap to slice-best = {row['mean_gap_to_slice_best_pp']:.3f} pp, "
            f"max gap = {row['max_gap_to_slice_best_pp']:.3f} pp, "
            f"mean slowdown vs fastest = {row['mean_slowdown_vs_fastest']:.3f}x"
        )
    lines.append("")
    lines.append("## Grouping at N=128")
    for _, row in group_safe_speedups.iterrows():
        lines.append(
            f"- {row['domain']} | w={int(row['width'])} | {ALGORITHM_LABELS[row['algorithm']]}: "
            f"best measured speedup at g={int(row['G'])} is {row['speedup_vs_G1']:.3f}x "
            f"with train delta {row['train_acc_delta_pp']:.3f} pp"
        )
    lines.append("")
    lines.append("## Best time-aware allocations (groups allowed)")
    for _, row in best_with_group[best_with_group["compute"] == HIGH_BUDGET].iterrows():
        lines.append(
            f"- {row['domain']}: {ALGORITHM_LABELS[row['algorithm']]}, "
            f"n={int(row['B'])}, k={int(row['K'])}, g={int(row['G'])}, T={int(row['iteration'])}, "
            f"w={int(row['width'])}, train={row['train_acc01_mean']:.4f}, test={row['test_acc01_mean']:.4f}, "
            f"time={row['wall_time_s_mean']:.1f}s"
        )

    (TEXT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
