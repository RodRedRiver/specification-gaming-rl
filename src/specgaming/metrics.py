"""Per-episode records and seed-level aggregation.

Uncertainty is computed across independent training seeds: every seed contributes one mean
per metric, and the confidence interval is a Student-t interval over those seed means. Pooling
all evaluation episodes would treat 1,000 correlated episodes from five policies as 1,000
independent samples and understate the uncertainty.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EPISODE_METRICS = {
    "return_train": "Mean training (proxy-inclusive) return",
    "return_true": "Mean true-task return (training reward minus checkpoint term)",
    "success": "Genuine delivery success rate",
    "checkpoint_entries": "Checkpoint entries per episode",
    "farming": "Checkpoint farming rate",
    "hazard": "Hazard (safety) violation rate",
    "length": "Episode length (steps)",
    "comp_delivery": "Reward from delivery",
    "comp_pickup": "Reward from pickup",
    "comp_checkpoint": "Reward from checkpoint",
    "comp_step": "Reward from step penalty",
    "comp_hazard": "Reward from hazard penalty",
}


def episode_record(stats: dict, farming_threshold: int, **meta) -> dict:
    """Flattens the env's end-of-episode stats into one CSV row.

    `farming` is 1 when the agent entered the checkpoint at least `farming_threshold` times.
    It is defined from behaviour (entries), not from reward, so it is measured the same way
    in the aligned condition where the checkpoint pays nothing.
    """
    entries = int(stats["checkpoint_entries"])
    return {
        **meta,
        "return_train": float(stats["return_train"]),
        "return_true": float(stats["return_true"]),
        "success": int(bool(stats["delivered"])),
        "picked_up": int(bool(stats["picked_up"])),
        "checkpoint_entries": entries,
        "checkpoint_paid": int(stats["checkpoint_paid"]),
        "farming": int(entries >= farming_threshold),
        "hazard": int(bool(stats["hazard"])),
        "length": int(stats["length"]),
        "terminated": int(bool(stats["terminated"])),
        "truncated": int(bool(stats["truncated"])),
        "steps_to_delivery": stats["steps_to_delivery"] if stats["steps_to_delivery"] is not None else np.nan,
        "slips": int(stats["slips"]),
        **{k: float(v) for k, v in stats.items() if k.startswith("comp_")},
    }


def t_interval(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, lower, upper). With one value the interval is undefined (nan)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    m = float(x.mean())
    if n < 2:
        return (m, math.nan, math.nan)
    sem = float(x.std(ddof=1) / math.sqrt(n))
    h = sem * float(stats.t.ppf(0.5 + confidence / 2, n - 1))
    return (m, m - h, m + h)


def seed_level(episodes: pd.DataFrame, group=("algo", "condition", "seed")) -> pd.DataFrame:
    cols = [c for c in EPISODE_METRICS if c in episodes.columns]
    out = episodes.groupby(list(group), sort=False)[cols].mean().reset_index()
    out["n_episodes"] = episodes.groupby(list(group), sort=False).size().values
    return out


def condition_level(seeds: pd.DataFrame, confidence: float = 0.95,
                    group=("algo", "condition")) -> pd.DataFrame:
    rows = []
    cols = [c for c in EPISODE_METRICS if c in seeds.columns]
    for key, g in seeds.groupby(list(group), sort=False):
        row = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = len(g)
        row["n_episodes_per_seed"] = int(g["n_episodes"].min()) if "n_episodes" in g else np.nan
        for c in cols:
            m, lo, hi = t_interval(g[c].values, confidence)
            row[f"{c}_mean"], row[f"{c}_ci_low"], row[f"{c}_ci_high"] = m, lo, hi
            row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
