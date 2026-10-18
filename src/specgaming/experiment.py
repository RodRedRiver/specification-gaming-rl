"""Training, evaluation and run bookkeeping."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import env_config_for
from .env import ENV_ID, ENV_VERSION, DeliveryGridWorld
from .metrics import episode_record
from .policies import SCRIPTED

PACKAGES = ("gymnasium", "stable-baselines3", "torch", "numpy", "pandas", "matplotlib", "scipy", "pytest")

# Eval seeds live far from training seeds so evaluation never reuses a training reset stream.
EVAL_SEED_OFFSET = 1_000_000


def package_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_dirty(cwd: Path) -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "scripts"],
                             cwd=cwd, capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_id(algo: str, condition: str, seed: int) -> str:
    return f"{algo}__{condition}__seed{seed}"


# ------------------------------------------------------------------ training
def _make_model(algo: str, env, seed: int, hp: dict, log_dir: Path):
    from stable_baselines3 import DQN, PPO

    kwargs = {k: v for k, v in hp.items() if k != "total_timesteps"}
    if "policy_kwargs" in kwargs and "net_arch" in kwargs["policy_kwargs"]:
        kwargs["policy_kwargs"] = dict(kwargs["policy_kwargs"])
    cls = {"ppo": PPO, "dqn": DQN}[algo]
    return cls("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **kwargs)


def train_one(cfg: dict, algo: str, condition: dict, seed: int, out_dir: Path) -> dict:
    """Trains one (algorithm, condition, seed) run and saves the model and Monitor log."""
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    rid = run_id(algo, condition["name"], seed)
    model_dir = out_dir / "models"
    log_dir = out_dir / "train_logs" / rid
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = env_config_for(cfg, condition)
    hp = dict(cfg["hyperparameters"][algo])
    n_envs = int(hp.pop("n_envs", 1))
    total = int(hp["total_timesteps"])

    def monitored(rank: int):
        def _f():
            env = DeliveryGridWorld(env_cfg)
            return Monitor(env, str(log_dir / f"env{rank}"),
                           info_keywords=("is_success", "hazard", "checkpoint_entries"))
        return _f

    vec = DummyVecEnv([monitored(i) for i in range(n_envs)])
    vec.seed(seed)  # env i is reset with seed + i

    t0 = time.time()
    model = _make_model(algo, vec, seed, hp, log_dir)
    model.learn(total_timesteps=total, progress_bar=False)
    model.save(model_dir / rid)
    vec.close()
    return {"run_id": rid, "algo": algo, "condition": condition["name"], "seed": seed,
            "total_timesteps": total, "n_envs": n_envs, "train_seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- evaluation
def rollout(env: DeliveryGridWorld, act, eval_seed: int, record_frames: bool = False):
    obs, _ = env.reset(seed=eval_seed)
    frames = [env.render()] if record_frames else None
    while True:
        obs, _, term, trunc, info = env.step(act(obs))
        if record_frames:
            frames.append(env.render())
        if term or trunc:
            return info["episode_stats"], list(env.trajectory), frames


def evaluate_policy(cfg: dict, condition: dict, act, n_episodes: int, meta: dict,
                    env: DeliveryGridWorld | None = None) -> tuple[list[dict], list[list]]:
    env = env or DeliveryGridWorld(env_config_for(cfg, condition))
    thr = int(cfg["eval"]["farming_threshold"])
    rows, trajs = [], []
    for ep in range(n_episodes):
        eval_seed = EVAL_SEED_OFFSET + ep
        stats, traj, _ = rollout(env, act, eval_seed)
        rows.append(episode_record(stats, thr, **meta, episode=ep, eval_seed=eval_seed,
                                   proxy_weight=float(condition["checkpoint_reward"]),
                                   checkpoint_once=int(bool(condition.get("checkpoint_once", False)))))
        trajs.append(traj)
    return rows, trajs


def evaluate_one(cfg: dict, algo: str, condition: dict, seed: int, out_dir: Path) -> dict:
    import torch
    from stable_baselines3 import DQN, PPO

    torch.set_num_threads(1)
    rid = run_id(algo, condition["name"], seed)
    cls = {"ppo": PPO, "dqn": DQN}[algo]
    model = cls.load(out_dir / "models" / f"{rid}.zip", device="cpu")
    deterministic = bool(cfg["eval"].get("deterministic", True))

    def act(obs):
        a, _ = model.predict(obs, deterministic=deterministic)
        return int(a)

    n = int(cfg["eval"]["episodes"])
    rows, trajs = evaluate_policy(cfg, condition, act, n,
                                  {"algo": algo, "condition": condition["name"], "seed": seed})
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    with open(traj_dir / f"{rid}.json", "w") as f:
        json.dump([[list(p) for p in t] for t in trajs], f)
    return {"run_id": rid, "rows": rows}


def evaluate_scripted(cfg: dict, n_episodes: int) -> pd.DataFrame:
    """Every scripted policy under every reward condition: what does each reward actually pay?"""
    rows = []
    for cond in cfg["conditions"]:
        env = DeliveryGridWorld(env_config_for(cfg, cond))
        for name, cls in SCRIPTED.items():
            pol = cls(env, seed=0) if name == "random" else cls(env)
            r, _ = evaluate_policy(cfg, cond, pol, n_episodes,
                                   {"algo": f"scripted:{name}", "condition": cond["name"], "seed": 0},
                                   env=env)
            rows.extend(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ manifest
def manifest(cfg: dict, root: Path, extra: dict | None = None) -> dict:
    env_cfgs = {c["name"]: env_config_for(cfg, c).to_dict() for c in cfg["conditions"]}
    return {
        "experiment": cfg["name"],
        "config_path": cfg.get("_path"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "env_id": ENV_ID,
        "env_version": ENV_VERSION,
        "env_configs": env_cfgs,
        "eval_seed_offset": EVAL_SEED_OFFSET,
        "versions": package_versions(),
        "git_commit": git_commit(root),
        "git_dirty_code": git_dirty(root),
        "cpu_count": os.cpu_count(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }


def safe_call(fn, *args):
    try:
        return {"ok": True, **fn(*args)}
    except Exception as e:  # recorded in the manifest as a failed run, never silently dropped
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                "args": [a if isinstance(a, (str, int, float)) else getattr(a, "get", lambda k: None)("name")
                         for a in args[1:4]]}


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x
