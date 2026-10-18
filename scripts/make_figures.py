#!/usr/bin/env python
"""Figures, trajectory heatmaps and a GIF of trained-agent behaviour from saved results.

    python scripts/make_figures.py --config configs/full.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specgaming import plotting as P  # noqa: E402
from specgaming.config import env_config_for, load_config  # noqa: E402
from specgaming.env import DeliveryGridWorld  # noqa: E402
from specgaming.experiment import EVAL_SEED_OFFSET, rollout, run_id  # noqa: E402
from specgaming.render import draw_grid  # noqa: E402


def load_tables(results: Path, cfg: dict):
    tag = "_".join(cfg["algorithms"])
    cond = pd.read_csv(results / "summary" / f"condition_level_{tag}.csv")
    seeds = pd.read_csv(results / "summary" / f"seed_level_{tag}.csv")
    weights = {c["name"]: float(c["checkpoint_reward"]) for c in cfg["conditions"]}
    for d in (cond, seeds):
        d["proxy_weight"] = d["condition"].map(weights)
    return cond, seeds


def make_gif(results: Path, cfg: dict, out: Path, algo: str = "ppo", seed: int | None = None,
             conds=("aligned", "strong", "mitigation"), episode: int = 0):
    """Side-by-side replay of real trained policies on the same evaluation seed."""
    from stable_baselines3 import DQN, PPO

    seed = cfg["seeds"][0] if seed is None else seed
    cls = {"ppo": PPO, "dqn": DQN}[algo]
    by_name = {c["name"]: c for c in cfg["conditions"]}
    clips = []
    for name in conds:
        model = cls.load(results / "models" / f"{run_id(algo, name, seed)}.zip", device="cpu")
        env = DeliveryGridWorld(env_config_for(cfg, by_name[name]), render_mode="rgb_array")
        frames = []

        def act(obs):
            return int(model.predict(obs, deterministic=True)[0])

        obs, _ = env.reset(seed=EVAL_SEED_OFFSET + episode)
        while True:
            st = env.stats
            frames.append(draw_grid(env, cell=40, caption=f"{name} t={env.t} cp={st['checkpoint_entries']}"
                                    f"{' DELIVERED' if st['delivered'] else ''}{' HAZARD' if st['hazard'] else ''}"))
            obs, _, term, trunc, _ = env.step(act(obs))
            if term or trunc:
                st = env.stats
                end = "DELIVERED" if st["delivered"] else ("HAZARD" if st["hazard"] else "TIME LIMIT")
                frames.append(draw_grid(env, cell=40, caption=f"{name} t={env.t} cp={st['checkpoint_entries']} {end}"))
                break
        clips.append(frames)
    n = max(len(c) for c in clips)
    sep = np.full((clips[0][0].shape[0], 6, 3), 255, dtype=np.uint8)
    combined = []
    for i in range(n):
        tiles = [c[min(i, len(c) - 1)] for c in clips]
        row = tiles[0]
        for t in tiles[1:]:
            row = np.concatenate([row, sep, t], axis=1)
        combined.append(row)
    combined += [combined[-1]] * 8
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"agent_behaviour_{algo}_seed{seed}.gif"
    imageio.mimsave(path, combined, duration=0.18, loop=0)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = ROOT / "results" / cfg["name"]
    out = ROOT / "figures" / cfg["name"]
    cond, seeds = load_tables(results, cfg)

    P.layout_diagram(DeliveryGridWorld(env_config_for(cfg, cfg["conditions"][0])), out)
    P.weight_curve(cond, "success", "Genuine delivery success (% of episodes)",
                   "Proxy reward weight vs true task success", out, "fig1_success_vs_proxy_weight")
    P.weight_curve(cond, "hazard", "Hazard-zone violations (% of episodes)",
                   "Proxy reward weight vs safety violations", out, "fig2_hazard_vs_proxy_weight")
    P.reported_vs_true(seeds, cond, out)
    P.exploitation(cond, out)
    P.mitigation(cond, out)
    curves = P.learning_curves(results, cfg, out)
    if not curves.empty:
        curves.to_csv(results / "summary" / "learning_curves.csv", index=False)
    P.decomposition(cond, out)

    def env_factory(name):
        return DeliveryGridWorld(env_config_for(cfg, next(c for c in cfg["conditions"] if c["name"] == name)))

    for algo in cfg["algorithms"]:
        P.visitation_heatmaps(results, cfg, algo, env_factory, out)
    print("gif:", make_gif(results, cfg, out, "ppo"))
    print("figures ->", out)


if __name__ == "__main__":
    main()
