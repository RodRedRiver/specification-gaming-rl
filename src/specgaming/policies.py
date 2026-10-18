"""Scripted reference policies.

They are not baselines to beat. They measure what each reward function actually pays for
qualitatively different behaviours (deliver, farm, farm-then-deliver), independent of whether
a learner finds them. This tells us whether the incentive to game exists before asking
whether PPO exploits it.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .env import ACTIONS, DeliveryGridWorld


def _bfs_next_action(env: DeliveryGridWorld, start, goal) -> int:
    """First action of a shortest hazard-free path from start to goal (ties broken by action order)."""
    if start == goal:
        return 0
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for a, (dr, dc) in ACTIONS.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < env.n_rows and 0 <= nxt[1] < env.n_cols):
                continue
            if nxt in env.hazards or nxt in prev:
                continue
            prev[nxt] = (cur, a)
            q.append(nxt)
    if goal not in prev:
        raise ValueError("goal unreachable without crossing the hazard")
    node = goal
    while prev[node][0] != start:
        node = prev[node][0]
    return prev[node][1]


def shortest_path_length(env: DeliveryGridWorld, start, goal) -> int:
    n, pos = 0, start
    while pos != goal:
        dr, dc = ACTIONS[_bfs_next_action(env, pos, goal)]
        pos = (pos[0] + dr, pos[1] + dc)
        n += 1
    return n


class ScriptedPolicy:
    name = "scripted"

    def __init__(self, env: DeliveryGridWorld):
        self.env = env

    def __call__(self, obs) -> int:
        raise NotImplementedError


class RandomPolicy(ScriptedPolicy):
    name = "random"

    def __init__(self, env, seed: int = 0):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return int(self.rng.integers(4))


class DeliverPolicy(ScriptedPolicy):
    """Intended behaviour: shortest hazard-free route to the package, then to the destination."""

    name = "deliver"

    def __call__(self, obs):
        e = self.env
        goal = e.destination if e.has_package else e.package_pos
        return _bfs_next_action(e, e.pos, goal)


class FarmPolicy(ScriptedPolicy):
    """Pure gaming: shuttle between the start cell and the checkpoint for the whole episode."""

    name = "farm"

    def __call__(self, obs):
        e = self.env
        anchor = e.start
        if e.pos == e.checkpoint:
            return _bfs_next_action(e, e.pos, anchor)
        return _bfs_next_action(e, e.pos, e.checkpoint)


class FarmThenDeliverPolicy(ScriptedPolicy):
    """Farm until only enough time remains to deliver (plus a margin), then deliver."""

    name = "farm_then_deliver"

    def __init__(self, env, margin: int = 6):
        super().__init__(env)
        self.margin = margin
        self.deliver = DeliverPolicy(env)
        self.farm = FarmPolicy(env)
        self.need = (shortest_path_length(env, env.start, env.package_pos)
                     + shortest_path_length(env, env.package_pos, env.destination))

    def __call__(self, obs):
        e = self.env
        remaining = e.config.max_steps - e.t
        if e.has_package or remaining <= self.need + self.margin:
            return self.deliver(obs)
        return self.farm(obs)


SCRIPTED = {p.name: p for p in (RandomPolicy, DeliverPolicy, FarmPolicy, FarmThenDeliverPolicy)}
