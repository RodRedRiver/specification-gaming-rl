#!/usr/bin/env python
"""Train and evaluate every (algorithm, condition, seed) run in a config.

    python scripts/run_experiment.py --config configs/smoke.yaml
    python scripts/run_experiment.py --config configs/full.yaml --algos ppo

Outputs go to results/<config name>/. Failed runs are recorded in runs.json and the
manifest; they are never dropped silently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specgaming.config import load_config  # noqa: E402
from specgaming.experiment import (  # noqa: E402
    evaluate_one, evaluate_scripted, manifest, run_id, safe_call, to_jsonable, train_one,
)
from specgaming.metrics import condition_level, seed_level  # noqa: E402


def _train(args):
    cfg, algo, cond, seed, out = args
    return safe_call(train_one, cfg, algo, cond, seed, out)


def _eval(args):
    cfg, algo, cond, seed, out = args
    return safe_call(evaluate_one, cfg, algo, cond, seed, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--algos", nargs="*", help="subset of the config's algorithms")
    ap.add_argument("--workers", type=int, help="override config workers")
    ap.add_argument("--skip-train", action="store_true", help="evaluate existing models only")
    a = ap.parse_args()

    cfg = load_config(a.config)
    algos = a.algos or cfg["algorithms"]
    workers = a.workers or int(cfg.get("workers", 1))
    out = ROOT / "results" / cfg["name"]
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "summary").mkdir(parents=True, exist_ok=True)

    jobs = [(cfg, algo, cond, seed, out) for algo in algos for cond in cfg["conditions"] for seed in cfg["seeds"]]
    print(f"{len(jobs)} runs: algos={algos} conditions={[c['name'] for c in cfg['conditions']]} "
          f"seeds={cfg['seeds']} workers={workers}", flush=True)
    t0 = time.time()
    runs_path = out / "runs.json"
    previous = json.loads(runs_path.read_text()) if runs_path.exists() else {}

    ctx = get_context("spawn")
    train_results = []
    if not a.skip_train:
        with ctx.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_train, jobs), 1):
                train_results.append(r)
                tag = r.get("run_id", r.get("args"))
                status = f"{r['train_seconds']}s" if r["ok"] else f"FAILED {r['error']}"
                print(f"[train {i}/{len(jobs)}] {tag}: {status}", flush=True)

    with ctx.Pool(workers) as pool:
        eval_results = list(pool.imap_unordered(_eval, jobs))
    rows = [row for r in eval_results if r["ok"] for row in r["rows"]]
    failed_eval = [r for r in eval_results if not r["ok"]]
    for r in failed_eval:
        print(f"[eval] FAILED {r['args']}: {r['error']}", flush=True)

    episodes = pd.DataFrame(rows)
    algo_tag = "_".join(algos)
    if not episodes.empty:
        episodes = episodes.sort_values(["algo", "condition", "seed", "episode"])
        episodes.to_csv(out / "raw" / f"episodes_{algo_tag}.csv", index=False)
        seeds = seed_level(episodes)
        seeds.to_csv(out / "summary" / f"seed_level_{algo_tag}.csv", index=False)
        condition_level(seeds, cfg["eval"]["confidence"]).to_csv(
            out / "summary" / f"condition_level_{algo_tag}.csv", index=False)

    scripted = evaluate_scripted(cfg, int(cfg["eval"]["scripted_episodes"]))
    scripted.to_csv(out / "raw" / "scripted_episodes.csv", index=False)
    s_seeds = seed_level(scripted)
    s_seeds.drop(columns=["seed"]).to_csv(out / "summary" / "scripted_summary.csv", index=False)

    runs = dict(previous)
    for r in train_results:
        key = r.get("run_id") or "|".join(map(str, r["args"]))
        runs[key] = {k: v for k, v in r.items() if k != "traceback"} | (
            {"traceback": r["traceback"]} if not r["ok"] else {})
    for r in failed_eval:
        runs["eval|" + "|".join(map(str, r["args"]))] = {"ok": False, "stage": "eval", "error": r["error"]}
    runs_path.write_text(json.dumps(to_jsonable(runs), indent=2))

    expected = {run_id(al, c["name"], s) for (_, al, c, s, _) in jobs}
    evaluated = {run_id(r["algo"], r["condition"], r["seed"]) for r in rows} if rows else set()
    man = manifest(cfg, ROOT, {
        "algorithms_run": algos,
        "runs_expected": len(expected),
        "runs_evaluated": len(evaluated),
        "runs_missing": sorted(expected - evaluated),
        "failed_runs": [k for k, v in runs.items() if not v.get("ok", True)],
        "wall_seconds": round(time.time() - t0, 1),
    })
    (out / f"manifest_{algo_tag}.json").write_text(json.dumps(to_jsonable(man), indent=2))
    print(f"done in {time.time() - t0:.0f}s; evaluated {len(evaluated)}/{len(expected)} runs -> {out}")
    return 0 if evaluated == expected else 1


if __name__ == "__main__":
    sys.exit(main())
