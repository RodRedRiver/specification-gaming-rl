#!/usr/bin/env python
"""Writes results_summary.md from the recorded CSVs only.

    python scripts/summarize.py --config configs/full.yaml

Hypothesis verdicts apply the rule fixed in report/preregistration.md: the direction must hold
for the condition means AND the two 95 % seed-level intervals must not overlap. The narrative
(fellowship description, CV bullets) is read from report/narrative.md, which is written by hand
from these tables; if the full results do not exist, every numerical section says PENDING.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specgaming.config import load_config  # noqa: E402

ORDER = ["aligned", "weak", "moderate", "strong", "mitigation"]


def fmt(r, m, pct=False, nd=2):
    s = 100 if pct else 1
    mean, lo, hi = r[f"{m}_mean"] * s, r[f"{m}_ci_low"] * s, r[f"{m}_ci_high"] * s
    unit = "%" if pct else ""
    if pd.isna(lo):
        return f"{mean:.{nd if not pct else 1}f}{unit}"
    d = 1 if pct else nd
    if pct:  # a rate cannot leave [0, 100]; the t-interval can, so the displayed bounds are clipped
        lo, hi = max(lo, 0.0), min(hi, 100.0)
    return f"{mean:.{d}f}{unit} [{lo:.{d}f}, {hi:.{d}f}]"


def table(cond: pd.DataFrame, algo: str, weights: dict) -> str:
    g = cond[cond["algo"] == algo].set_index("condition")
    lines = ["| Condition | Proxy w | Seeds | Eps/seed | Training return | True return | Delivery success | "
             "Farming | Checkpoint entries | Hazard violations | Length |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in ORDER:
        if c not in g.index:
            continue
        r = g.loc[c]
        lines.append(f"| {c} | {weights[c]:.2f}{' once' if c == 'mitigation' else ''} | {int(r['n_seeds'])} | "
                     f"{int(r['n_episodes_per_seed'])} | {fmt(r, 'return_train')} | {fmt(r, 'return_true')} | "
                     f"{fmt(r, 'success', True)} | {fmt(r, 'farming', True)} | {fmt(r, 'checkpoint_entries', nd=1)} | "
                     f"{fmt(r, 'hazard', True)} | {fmt(r, 'length', nd=1)} |")
    return "\n".join(lines)


def seed_table(seeds: pd.DataFrame, algo: str) -> str:
    g = seeds[seeds["algo"] == algo]
    lines = ["| Condition | Seed | Success | Farming | Hazard | Training return |", "|---|---|---|---|---|---|"]
    for c in ORDER:
        for _, r in g[g["condition"] == c].sort_values("seed").iterrows():
            lines.append(f"| {c} | {int(r['seed'])} | {100 * r['success']:.1f}% | {100 * r['farming']:.1f}% | "
                         f"{100 * r['hazard']:.1f}% | {r['return_train']:.2f} |")
    return "\n".join(lines)


def learned_table(seeds: pd.DataFrame, algo: str, threshold: float = 0.5) -> str:
    """Success is bimodal across seeds, so also count seeds whose policy delivers in most episodes."""
    g = seeds[seeds["algo"] == algo]
    lines = ["| Condition | Seeds delivering in > 50% of episodes | Seeds farming in > 50% of episodes | "
             "Hazard rate, Wilson 95% with n = episodes per seed |", "|---|---|---|---|"]
    for c in ORDER:
        s = g[g["condition"] == c]
        if s.empty:
            continue
        # The training seeds share the same evaluation seeds, so pooling them does not add independent
        # episodes; use the per-seed episode count as n.
        n_ep = int(s["n_episodes"].min())
        k = int(round(s["hazard"].mean() * n_ep))
        z = 1.959964
        ph = k / n_ep
        centre = (ph + z * z / (2 * n_ep)) / (1 + z * z / n_ep)
        half = z * ((ph * (1 - ph) / n_ep + z * z / (4 * n_ep * n_ep)) ** 0.5) / (1 + z * z / n_ep)
        lines.append(f"| {c} | {int((s['success'] > threshold).sum())} / {len(s)} | "
                     f"{int((s['farming'] > threshold).sum())} / {len(s)} | "
                     f"{100 * ph:.1f}% [{100 * (centre - half):.1f}, {100 * (centre + half):.1f}] (n = {n_ep}) |")
    return "\n".join(lines)


def separated(a, b, m, direction):
    """direction '>' means a > b. Returns (direction holds, intervals do not overlap)."""
    holds = a[f"{m}_mean"] > b[f"{m}_mean"] if direction == ">" else a[f"{m}_mean"] < b[f"{m}_mean"]
    if direction == ">":
        apart = a[f"{m}_ci_low"] > b[f"{m}_ci_high"]
    else:
        apart = a[f"{m}_ci_high"] < b[f"{m}_ci_low"]
    return bool(holds), bool(apart)


def verdict(parts) -> str:
    if all(h and s for h, s in parts):
        return "**supported**"
    if all(h for h, _ in parts):
        return "**direction holds, not distinguishable at n = 5** (intervals overlap)"
    return "**not supported**"


def hypotheses(cond: pd.DataFrame, algo: str = "ppo") -> str:
    g = cond[cond["algo"] == algo].set_index("condition")
    if not set(ORDER) <= set(g.index):
        return "PENDING: not all conditions were evaluated."
    A, W, M, S, X = (g.loc[c] for c in ORDER)
    out = []

    lo_ok = A["farming_ci_high"] < 0.5 and W["farming_ci_high"] < 0.5
    hi_ok = M["farming_mean"] > 0.5 and S["farming_mean"] > 0.5
    hi_sep = M["farming_ci_low"] > 0.5 and S["farming_ci_low"] > 0.5
    h1 = "**supported**" if (lo_ok and hi_ok and hi_sep) else (
        "**direction holds, not distinguishable at n = 5**" if (A["farming_mean"] < 0.5 and W["farming_mean"] < 0.5 and hi_ok)
        else "**not supported**")
    out.append(f"* **H1 (gaming)**: farming aligned {fmt(A, 'farming', True)}, weak {fmt(W, 'farming', True)}, "
               f"moderate {fmt(M, 'farming', True)}, strong {fmt(S, 'farming', True)}. Verdict: {h1}.")

    p1 = separated(S, A, "return_train", ">")
    p2 = separated(S, A, "success", "<")
    out.append(f"* **H2 (Goodhart divergence)**: training return aligned {fmt(A, 'return_train')} vs strong "
               f"{fmt(S, 'return_train')}; delivery success aligned {fmt(A, 'success', True)} vs strong "
               f"{fmt(S, 'success', True)}. Verdict: {verdict([p1, p2])}.")

    gap = (A["success_mean"] - X["success_mean"]) * 100
    h3 = "**supported**" if (gap <= 10 and X["farming_ci_high"] < 0.10) else (
        "**direction holds, not distinguishable at n = 5**" if (gap <= 10 and X["farming_mean"] < 0.10) else "**not supported**")
    out.append(f"* **H3 (mitigation)**: success aligned {fmt(A, 'success', True)} vs mitigation "
               f"{fmt(X, 'success', True)} (gap {gap:.1f} points, threshold 10); farming under mitigation "
               f"{fmt(X, 'farming', True)} (threshold 10%). Verdict: {h3}.")

    q1 = separated(M, A, "hazard", ">")
    q2 = separated(S, A, "hazard", ">")
    out.append(f"* **H4 (safety side effect)**: hazard aligned {fmt(A, 'hazard', True)}, moderate "
               f"{fmt(M, 'hazard', True)}, strong {fmt(S, 'hazard', True)}. Verdict: {verdict([q1, q2])}.")
    return "\n".join(out)


def scripted_table(path: Path) -> str:
    d = pd.read_csv(path)
    lines = ["| Condition | Policy | Training return | True return | Success | Farming | Hazard |",
             "|---|---|---|---|---|---|---|"]
    for c in ORDER:
        for _, r in d[d["condition"] == c].iterrows():
            lines.append(f"| {c} | {r['algo'].split(':')[1]} | {r['return_train']:.2f} | {r['return_true']:.2f} | "
                         f"{100 * r['success']:.1f}% | {100 * r['farming']:.1f}% | {100 * r['hazard']:.1f}% |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--out", default="results_summary.md")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    res = ROOT / "results" / cfg["name"]
    tag = "_".join(cfg["algorithms"])
    weights = {c["name"]: float(c["checkpoint_reward"]) for c in cfg["conditions"]}
    narrative = ROOT / "report" / "narrative.md"
    cond_path = res / "summary" / f"condition_level_{tag}.csv"

    parts = ["# Results summary", ""]
    if not cond_path.exists():
        parts += ["**Status: pending experimental validation.** The full experiment has not been run, so this "
                  "file contains no findings. Run `python scripts/run_experiment.py --config configs/full.yaml`.",
                  "", "## Fellowship description", "", "PENDING", "", "## CV bullets", "", "PENDING"]
        (ROOT / a.out).write_text("\n".join(parts) + "\n")
        return

    man = json.loads((res / f"manifest_{tag}.json").read_text())
    cond = pd.read_csv(cond_path)
    seeds = pd.read_csv(res / "summary" / f"seed_level_{tag}.csv")
    hp = cfg["hyperparameters"]
    parts += [
        f"Generated by `scripts/summarize.py` from `results/{cfg['name']}/` (config `{a.config}`, git commit "
        f"`{(man.get('git_commit') or 'unknown')[:7]}`, run created {man['created_utc']}).",
        "",
        f"Runs evaluated: **{man['runs_evaluated']} of {man['runs_expected']}**. Failed runs: "
        f"{', '.join(man['failed_runs']) or 'none'}. Missing runs: {', '.join(man['runs_missing']) or 'none'}.",
        "",
        "Budgets: " + ", ".join(f"{k.upper()} {int(v['total_timesteps']):,} timesteps" for k, v in hp.items()
                                if k in cfg["algorithms"]) +
        f" per run; seeds {cfg['seeds']}; {cfg['eval']['episodes']} deterministic evaluation episodes per run. "
        "Brackets are 95% Student-t intervals over the per-seed means (n = number of seeds), not over pooled episodes; "
        "for rates the displayed bounds are clipped to [0, 100]%.",
        "",
    ]
    if narrative.exists():
        parts += [narrative.read_text().strip(), ""]
    else:
        parts += ["## Fellowship description", "", "PENDING: written by hand from the tables below.", "",
                  "## CV bullets", "", "PENDING", ""]
    if cfg["name"] == "full":
        parts += ["## Preregistered hypotheses (PPO, rule from report/preregistration.md)", ""]
    else:
        parts += ["## Hypotheses re-checked on POST-HOC data (PPO)", "",
                  f"`{cfg['name']}` was designed after seeing the main results and is not preregistered. The "
                  "preregistered rule is applied here only for comparison with the main run.", ""]
    parts += [hypotheses(cond, "ppo"), ""]
    for algo in [x for x in ("ppo", "dqn") if x in set(cond["algo"])]:
        parts += [f"## {algo.upper()}: condition means", "", table(cond, algo, weights), "",
                  f"### {algo.upper()}: seed counts and episode-level hazard interval", "",
                  "The seed-level t-interval above captures variation between training seeds only: every seed "
                  "is evaluated on the same evaluation seeds, so when all seeds learn the same policy the interval "
                  "collapses. The Wilson interval shows the complementary episode-sampling uncertainty of the "
                  "seed-mean hazard rate with n = episodes per seed, because the training seeds share evaluation "
                  "seeds and pooling them would overstate the sample size.", "", learned_table(seeds, algo), ""]
    parts += ["## Scripted reference policies (what each reward pays; not learners)", "",
              f"{cfg['eval']['scripted_episodes']} episodes each, slips on.", "",
              scripted_table(res / "summary" / "scripted_summary.csv"), ""]
    for algo in [x for x in ("ppo", "dqn") if x in set(cond["algo"])]:
        parts += [f"## {algo.upper()}: every seed (no seed omitted)", "", seed_table(seeds, algo), ""]
    (ROOT / a.out).write_text("\n".join(parts) + "\n")
    print("wrote", ROOT / a.out)


if __name__ == "__main__":
    main()
