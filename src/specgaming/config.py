"""Experiment configuration: YAML -> plain dicts plus typed env configs per reward condition."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .env import EnvConfig, RewardConfig

REQUIRED_KEYS = ("name", "algorithms", "conditions", "seeds", "env", "eval", "hyperparameters")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _read(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    return _merge(_read(path.parent / parent), cfg) if parent else cfg


def load_config(path: str | Path) -> dict:
    cfg = _read(Path(path))
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config {path} is missing keys: {missing}")
    for algo in cfg["algorithms"]:
        if algo not in cfg["hyperparameters"]:
            raise ValueError(f"no hyperparameters for algorithm {algo!r}")
        if "total_timesteps" not in cfg["hyperparameters"][algo]:
            raise ValueError(f"hyperparameters.{algo}.total_timesteps is required")
    names = [c["name"] for c in cfg["conditions"]]
    if len(set(names)) != len(names):
        raise ValueError("condition names must be unique")
    cfg["_path"] = str(path)
    return cfg


def env_config_for(cfg: dict, condition: dict) -> EnvConfig:
    """Builds the EnvConfig for one condition. Everything except the checkpoint terms comes from cfg['env']."""
    env = copy.deepcopy(cfg["env"])
    base_reward = env.pop("reward", {})
    reward = RewardConfig(
        **base_reward,
        checkpoint_reward=float(condition["checkpoint_reward"]),
        checkpoint_once=bool(condition.get("checkpoint_once", False)),
    )
    if "layout" in env:
        env["layout"] = tuple(env["layout"])
    return EnvConfig(**env, reward=reward)
