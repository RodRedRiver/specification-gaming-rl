from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from specgaming.config import env_config_for, load_config
from specgaming.env import DeliveryGridWorld, EnvConfig, RewardConfig
from specgaming.experiment import evaluate_policy
from specgaming.metrics import condition_level, episode_record, seed_level, t_interval
from specgaming.policies import DeliverPolicy, FarmPolicy, FarmThenDeliverPolicy, shortest_path_length

ROOT = Path(__file__).resolve().parents[1]


def _stats(**over):
    base = {"checkpoint_entries": 0, "checkpoint_paid": 0, "picked_up": False, "delivered": False,
            "hazard": False, "steps_to_pickup": None, "steps_to_delivery": None, "slips": 0,
            "return_train": 0.0, "return_true": 0.0, "length": 100, "terminated": False,
            "truncated": True, "comp_delivery": 0.0, "comp_pickup": 0.0, "comp_checkpoint": 0.0,
            "comp_step": -1.0, "comp_hazard": 0.0}
    return base | over


def test_farming_flag_uses_threshold_on_entries():
    assert episode_record(_stats(checkpoint_entries=2), 3)["farming"] == 0
    assert episode_record(_stats(checkpoint_entries=3), 3)["farming"] == 1


def test_t_interval_matches_scipy():
    x = [0.2, 0.4, 0.5, 0.9, 0.7]
    m, lo, hi = t_interval(x)
    ref = stats.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=stats.sem(x))
    assert m == pytest.approx(np.mean(x))
    assert (lo, hi) == pytest.approx(ref)


def test_t_interval_single_value_is_undefined():
    m, lo, hi = t_interval([0.5])
    assert m == 0.5 and np.isnan(lo) and np.isnan(hi)


def test_ci_is_across_seeds_not_episodes():
    # Two seeds, one always succeeds, one never does: the interval must be wide even with
    # many episodes per seed. A pooled-episode interval would be near zero width.
    rows = [{"algo": "x", "condition": "c", "seed": s, "success": float(s == 0)}
            for s in (0, 1) for _ in range(1000)]
    seeds = seed_level(pd.DataFrame(rows))
    assert list(seeds["n_episodes"]) == [1000, 1000]
    c = condition_level(seeds).iloc[0]
    assert c["n_seeds"] == 2 and c["success_mean"] == pytest.approx(0.5)
    assert c["success_ci_high"] - c["success_ci_low"] > 5


def test_deliver_policy_takes_shortest_safe_route():
    env = DeliveryGridWorld(EnvConfig(slip_prob=0.0))
    rows, _ = evaluate_policy({"eval": {"farming_threshold": 3}}, {"name": "aligned", "checkpoint_reward": 0.0},
                              DeliverPolicy(env), 3, {"algo": "deliver", "condition": "aligned", "seed": 0}, env=env)
    for r in rows:
        assert r["success"] == 1 and r["hazard"] == 0 and r["length"] == 15


def test_shortest_path_lengths():
    env = DeliveryGridWorld(EnvConfig())
    assert shortest_path_length(env, env.start, env.package_pos) == 9
    assert shortest_path_length(env, env.package_pos, env.destination) == 6


def _scripted_return(policy_cls, w, once=False, n=1):
    env = DeliveryGridWorld(EnvConfig(slip_prob=0.0, reward=RewardConfig(checkpoint_reward=w, checkpoint_once=once)))
    pol = policy_cls(env)
    rows, _ = evaluate_policy({"eval": {"farming_threshold": 3}},
                              {"name": "c", "checkpoint_reward": w, "checkpoint_once": once},
                              pol, n, {"algo": "s", "condition": "c", "seed": 0}, env=env)
    return rows[0]


def test_farm_policy_return_matches_hand_calculation():
    r = _scripted_return(FarmPolicy, 0.3)
    assert r["checkpoint_entries"] == 50 and r["length"] == 100 and r["success"] == 0
    assert r["return_train"] == pytest.approx(50 * 0.3 - 1.0)
    assert r["return_true"] == pytest.approx(-1.0)


def test_incentive_ordering_without_slip():
    deliver = _scripted_return(DeliverPolicy, 0.3)["return_train"]
    farm = _scripted_return(FarmPolicy, 0.3)["return_train"]
    assert farm > deliver  # strong proxy: gaming pays more than the task
    assert _scripted_return(FarmPolicy, 0.0)["return_train"] < _scripted_return(DeliverPolicy, 0.0)["return_train"]
    # mitigation: farming no longer pays more than delivering
    assert _scripted_return(FarmPolicy, 0.3, once=True)["return_train"] < \
        _scripted_return(DeliverPolicy, 0.3, once=True)["return_train"]


def test_farm_then_deliver_still_delivers():
    r = _scripted_return(FarmThenDeliverPolicy, 0.3)
    assert r["success"] == 1 and r["farming"] == 1


def test_configs_load_and_only_checkpoint_terms_differ():
    for name in ("smoke", "full"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        envs = [env_config_for(cfg, c).to_dict() for c in cfg["conditions"]]
        for e in envs:
            e["reward"].pop("checkpoint_reward")
            e["reward"].pop("checkpoint_once")
        assert all(e == envs[0] for e in envs)
    full = load_config(ROOT / "configs" / "full.yaml")
    assert len(full["seeds"]) >= 5 and full["eval"]["episodes"] >= 200
