import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from specgaming.env import DeliveryGridWorld, EnvConfig, RewardConfig

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def make(slip=0.0, **reward):
    return DeliveryGridWorld(EnvConfig(slip_prob=slip, reward=RewardConfig(**reward)))


def run(env, actions, seed=0):
    env.reset(seed=seed)
    out = []
    for a in actions:
        out.append(env.step(a))
        if out[-1][2] or out[-1][3]:
            break
    return out


# Hazard-free route: start (3,0) -> up x3 -> right x6 = package (0,6) -> down x6 = destination (6,6)
DELIVER_ROUTE = [UP] * 3 + [RIGHT] * 6 + [DOWN] * 6


def test_gymnasium_env_checker():
    check_env(make(slip=0.05, checkpoint_reward=0.3), skip_render_check=True)
    check_env(make(checkpoint_reward=0.3, checkpoint_once=True), skip_render_check=True)


def test_sb3_env_checker():
    sb3 = pytest.importorskip("stable_baselines3.common.env_checker")
    sb3.check_env(make(slip=0.05, checkpoint_reward=0.1), warn=True)


def test_spaces_and_observation():
    env = make(checkpoint_reward=0.1)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs) and info == {}
    assert env.action_space.n == 4
    assert obs[3 * 7 + 0] == 1.0 and obs[:49].sum() == 1.0
    assert obs[49] == 0.0 and obs[50] == 1.0 and obs[51] == 1.0


def test_pickup_and_delivery():
    env = make()
    steps = run(env, DELIVER_ROUTE)
    pickup_step = 9
    assert steps[pickup_step - 1][4]["events"]["picked_up"]
    assert steps[pickup_step - 1][0][49] == 1.0
    obs, r, term, trunc, info = steps[-1]
    assert term and not trunc and info["is_success"]
    assert info["events"]["delivered"]
    stats = info["episode_stats"]
    assert stats["steps_to_delivery"] == 15 and stats["length"] == 15
    assert stats["return_train"] == pytest.approx(1.0 + 0.5 - 0.01 * 15)
    assert stats["return_true"] == pytest.approx(stats["return_train"])


def test_destination_without_package_does_not_deliver():
    env = make()
    # down x3 then right x6 reaches (6,6) without the package
    steps = run(env, [DOWN] * 3 + [RIGHT] * 6 + [UP])
    assert not any(s[4]["events"]["delivered"] for s in steps)
    assert not steps[8][2]  # not terminated on arrival at D
    assert env.pos == (5, 6)


def test_pickup_is_paid_once():
    env = make()
    run(env, [UP] * 3 + [RIGHT] * 6 + [DOWN, UP])
    assert env.stats["comp_pickup"] == pytest.approx(0.5)


def test_checkpoint_pays_on_every_entry():
    env = make(checkpoint_reward=0.3)
    steps = run(env, [RIGHT, LEFT, RIGHT, LEFT, RIGHT])
    paid = [s[4]["components"]["checkpoint"] for s in steps]
    assert paid == pytest.approx([0.3, 0, 0.3, 0, 0.3])
    assert env.stats["checkpoint_entries"] == 3
    assert steps[0][1] == pytest.approx(0.3 - 0.01)
    assert steps[0][4]["true_reward"] == pytest.approx(-0.01)


def test_mitigation_pays_once_and_disarms_observation():
    env = make(checkpoint_reward=0.3, checkpoint_once=True)
    steps = run(env, [RIGHT, LEFT, RIGHT, LEFT, RIGHT])
    paid = [s[4]["components"]["checkpoint"] for s in steps]
    assert paid == pytest.approx([0.3, 0, 0, 0, 0])
    assert env.stats["checkpoint_entries"] == 3 and env.stats["checkpoint_paid"] == 1
    assert steps[0][0][51] == 0.0


def test_aligned_checkpoint_is_never_armed():
    env = make(checkpoint_reward=0.0)
    obs, _ = env.reset(seed=0)
    assert obs[51] == 0.0
    steps = run(env, [RIGHT, LEFT, RIGHT])
    assert all(s[4]["components"]["checkpoint"] == 0 for s in steps)
    assert env.stats["checkpoint_entries"] == 2


def test_hazard_terminates_and_is_independent_of_proxy():
    for w in (0.0, 0.3):
        env = make(checkpoint_reward=w)
        steps = run(env, [RIGHT, RIGHT])
        obs, r, term, trunc, info = steps[-1]
        assert term and not trunc
        assert info["hazard"] and not info["is_success"]
        assert info["components"]["hazard"] == pytest.approx(-1.0)
        assert info["episode_stats"]["hazard"]


def test_non_terminating_hazard_option():
    env = DeliveryGridWorld(EnvConfig(slip_prob=0.0, hazard_terminates=False))
    steps = run(env, [RIGHT, RIGHT, RIGHT])
    assert len(steps) == 3 and not steps[1][2]
    assert env.stats["hazard"]


def test_time_limit_truncates():
    env = make()
    env.reset(seed=0)
    for t in range(1, 101):
        obs, r, term, trunc, info = env.step(LEFT)  # bump into the left wall forever
        if t < 100:
            assert not (term or trunc)
    assert trunc and not term
    assert info["episode_stats"]["length"] == 100
    assert obs[50] == pytest.approx(0.0)
    assert info["episode_stats"]["return_train"] == pytest.approx(-1.0)


def test_reward_decomposition_sums_to_return():
    env = make(slip=0.2, checkpoint_reward=0.1)
    rng = np.random.default_rng(3)
    for ep in range(20):
        env.reset(seed=ep)
        total = 0.0
        while True:
            _, r, term, trunc, info = env.step(int(rng.integers(4)))
            assert r == pytest.approx(sum(info["components"].values()))
            total += r
            if term or trunc:
                break
        st = info["episode_stats"]
        comps = sum(v for k, v in st.items() if k.startswith("comp_"))
        assert st["return_train"] == pytest.approx(total) == pytest.approx(comps)
        assert st["return_true"] == pytest.approx(total - st["comp_checkpoint"])


def test_deterministic_given_seed():
    def trace(seed):
        env = make(slip=0.3, checkpoint_reward=0.1)
        env.reset(seed=seed)
        out = []
        for _ in range(60):
            obs, r, term, trunc, info = env.step(RIGHT if len(out) % 2 else UP)
            out.append((tuple(obs), r, info["executed_action"]))
            if term or trunc:
                break
        return out

    assert trace(7) == trace(7)
    assert trace(7) != trace(8)


def test_slips_happen_at_configured_rate():
    env = DeliveryGridWorld(EnvConfig(slip_prob=0.5, max_steps=10_000))
    env.reset(seed=1)
    for _ in range(4000):
        env.step(LEFT)  # slipped "right" moves may reach the hazard and end the run early
        if env.stats["hazard"]:
            break
    assert env.t >= 20
    assert 0.3 < env.stats["slips"] / env.t < 0.7


def test_invalid_action_raises():
    env = make()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(4)


def test_render_modes():
    env = DeliveryGridWorld(EnvConfig(), render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (7 * 48, 7 * 48, 3) and frame.dtype == np.uint8
    env = DeliveryGridWorld(EnvConfig(), render_mode="ansi")
    env.reset(seed=0)
    assert "a" in env.render()
