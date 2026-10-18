# When Reward Optimization Goes Wrong: Specification Gaming in Reinforcement Learning

*Technical report. Every number here comes from `results/full/` (main experiment, preregistered) or
`results/followup_entropy/` (post-hoc follow-up). The full tables are in `results_summary.md` and
`results_followup_entropy.md`.*

## Abstract

We train PPO and DQN agents on a package-delivery GridWorld whose reward includes a checkpoint term paid on
every entry, at four weights (0, 0.03, 0.10, 0.30), plus a mitigation that pays 0.30 once per episode. We
evaluate them against the intended objective (delivery), a behavioural measure of gaming (farming the
checkpoint) and a safety measure the reward never mentions (entering a hazard). 5 seeds per condition,
300k timesteps, 200 evaluation episodes per seed.

With any repeatable checkpoint reward, all 15 PPO runs converged to the same policy: shuttle in and out of the
checkpoint. Delivery success was 0 %, and 43.6 % of episodes ended in the hazard, against 1.1 % under the
aligned reward. That includes the weak weight, where scripted rollouts show farming pays *less* than
delivering (−0.08 vs 1.34). This contradicts our preregistered hypothesis that a weak proxy would not be
exploited. The training logs point to an exploration failure: under the aligned reward, 3 of 5 PPO seeds
never discovered delivery at all and idled against a wall. A small repeatable reward reachable in one step
then beats idling, so PPO settles on it. DQN, which explores with ε-greedy, learned delivery on 5 of 5
aligned seeds and was much less captured by the weak proxy (1 of 5 seeds delivering, 8 % farming), but
farmed under moderate (80 %) and strong (97 %) weights. Paying the checkpoint once removed farming
(≤ 0.2 %) and restored hazard rates to 1.5–1.8 %. Its effect on delivery success could not be resolved at
n = 5 because success is bimodal across seeds.

## 1. Question

Specification gaming is when an optimiser satisfies the stated objective in a way its designer did not
intend (Krakovna et al., 2020). It is a concrete case of reward misspecification (Amodei et al., 2016) and of
Goodhart's law (Manheim & Garrabrant, 2018). We ask a narrow, measurable version: *as a repeatable proxy
reward grows, how do task success, gaming behaviour and an unrelated safety measure change, and does
making the proxy non-repeatable fix it?*

## 2. Method

**Environment.** `DeliveryGridWorld-v1`, a 7 × 7 grid (see README). The agent starts at S, must pick up a
package at the top right and deliver it to the bottom right (15 steps on the shortest safe route). A
checkpoint C sits next to S and borders a 3 × 3 hazard. Actions slip to a uniformly random action with
probability 0.05. Entering the hazard ends the episode. The horizon is 100 steps (truncation).

**Rewards.** Delivery +1, pickup +0.5, step −0.01, hazard −1 in every condition. The checkpoint pays
0 / 0.03 / 0.10 / 0.30 per entry (aligned / weak / moderate / strong), or 0.30 once (mitigation).

**Separating the three things measured.** *Success* is delivery, the intended objective. *Farming* is
≥ 3 checkpoint entries in an episode: a behavioural definition, measured the same way when the checkpoint
pays nothing. *Safety violation* is entering the hazard, defined from the state. These are reported
separately. Farming is gaming, and the hazard is a side effect, not part of the gaming definition.

**Incentive check.** Four scripted policies (deliver, farm, farm-then-deliver, random) were rolled out under
each reward (500 episodes) to measure what each reward actually pays, independent of learning:

| Training return | aligned | weak | moderate | strong | mitigation |
|---|---|---|---|---|---|
| deliver | 1.34 | 1.34 | 1.34 | 1.35 | 1.35 |
| farm | −1.19 | −0.08 | 2.50 | 9.87 | −0.89 |
| farm then deliver | −0.18 | 0.74 | 2.91 | 9.08 | 0.12 |

So gaming is *not* optimal under weak (delivering is best) and *is* optimal under moderate and strong. The
scripted farmer enters the hazard in 42.6 % of episodes because of slips next to the hazard.

**Training and evaluation.** Stable-Baselines3 PPO (primary) and DQN (secondary, untuned), one fixed
hyperparameter set each (`configs/base.yaml`), seeds 0–4, 300,000 timesteps. Each trained policy is
evaluated greedily for 200 episodes on evaluation seeds disjoint from training. Intervals are 95 %
Student-t over the 5 per-seed means.

**Preregistration.** Hypotheses H1–H4, the metrics and the decision rule were committed before the full run
(`report/preregistration.md`, commit `800ba5a`).

## 3. Results (main experiment, preregistered)

### PPO

| Condition | Success | Seeds delivering | Farming | Hazard | Training return | True return |
|---|---|---|---|---|---|---|
| aligned (0) | 39.0 % [0, 100] | 2 / 5 | 0.0 % | 1.1 % [0.0, 3.8] | −0.10 | −0.10 |
| weak (0.03) | 0.0 % | 0 / 5 | 98.0 % | 43.6 % [43.3, 43.9] | −0.10 | −1.18 |
| moderate (0.10) | 0.0 % | 0 / 5 | 98.0 % | 43.6 % [43.3, 43.9] | 2.42 | −1.18 |
| strong (0.30) | 0.0 % | 0 / 5 | 98.0 % | 43.6 % [43.3, 43.9] | 9.65 | −1.18 |
| mitigation (0.30 once) | 59.1 % [0, 100] | 3 / 5 | 0.2 % | 1.5 % [1.1, 1.9] | 0.75 | 0.45 |

* **All 15 PPO runs with a repeatable checkpoint learned the same policy**: shuttle between S and C. The
  seed-0 evaluation trajectories under weak and strong are identical in 199 of 200 episodes. Because every
  run is evaluated on the same 200 slip sequences, identical policies give near-identical numbers. The
  very narrow intervals mean "every seed learned this", not "the hazard rate is known to ±0.3 points".
  The episode-sampling uncertainty of a 43.5 % rate over 200 episodes is about [36.8, 50.4] (Wilson).
* **Reported reward and true objective come apart completely.** Training return rose from −0.10 (aligned)
  to 9.65 (strong) while success fell from 39 % to 0 %. The true return under every proxy condition was
  −1.18, lower than the aligned agents' −0.10.
* **Success is bimodal.** Each aligned and mitigation seed either delivers in ≥ 95 % of episodes or almost
  never (the rest time out at 100 steps). A t-interval over {100, 95, 0, 0, 0} spans the whole range. The
  seed counts are the more informative statistic.

### DQN (secondary)

| Condition | Success | Seeds delivering | Farming | Hazard | Training return |
|---|---|---|---|---|---|
| aligned | 96.4 % [89.1, 100] | 5 / 5 | 0.0 % | 1.0 % | 1.17 |
| weak | 43.4 % [4.3, 82.5] | 1 / 5 | 8.1 % [0, 25.6] | 5.3 % [0, 14.6] | −0.04 |
| moderate | 0.0 % | 0 / 5 | 80.4 % [34.3, 100] | 56.2 % [17.9, 94.5] | 1.00 |
| strong | 0.0 % | 0 / 5 | 97.4 % [96.7, 98.1] | 56.5 % [40.5, 72.5] | 8.25 |
| mitigation | 50.1 % [0, 100] | 2 / 5 | 0.0 % | 1.8 % [0.1, 3.5] | 0.43 |

DQN found the task on every aligned seed. Under the weak proxy most DQN episodes neither delivered nor
farmed (51 % timed out). The weak proxy degraded learning without being exploited as such.

### Preregistered hypotheses (PPO, mechanical rule)

* **H1, gaming only at moderate and strong weights: not supported.** Farming was 98 % under the weak weight too.
* **H2, Goodhart divergence: direction holds, not distinguishable at n = 5.** Training return rose and success
  fell, but the aligned success interval spans [0, 100] because only 2 of 5 aligned seeds learned the task.
* **H3, mitigation: supported by the preregistered rule**, with a caveat. Farming fell to 0.2 % and mean success
  (59 %) was not more than 10 points below aligned (39 %), but both success intervals span [0, 100]. We do
  not claim the mitigation *improves* success. DQN shows the opposite ordering (aligned 5/5 seeds,
  mitigation 2/5).
* **H4, safety side effect: supported.** Hazard 43.6 % under moderate and strong vs 1.1 % aligned, with
  non-overlapping intervals. The same holds for DQN (56 % vs 1 %).

## 4. Post-hoc follow-up: is it exploration? (not preregistered)

**Why it was run.** The main run's training logs showed 3 of 5 aligned PPO seeds never discovering
delivery. If PPO cannot find the task even without a proxy, the weak-proxy result could just mean that
*PPO with these settings fails at this task*. The follow-up (`configs/followup_entropy.yaml`) changes exactly one
setting, identically for all conditions: PPO entropy coefficient 0.01 → 0.05. It was designed after seeing
the main results, so it is exploratory.

| PPO, entropy 0.05 | Success | Seeds delivering | Farming | Hazard | Training return |
|---|---|---|---|---|---|
| aligned | 79.9 % [24.4, 100] | 4 / 5 | 0.0 % | 0.0 % | 0.86 |
| weak | 0.0 % | 0 / 5 | 97.8 % | 49.0 % [34.4, 63.6] | −0.17 |
| moderate | 0.0 % | 0 / 5 | 98.0 % | 43.7 % | 2.43 |
| strong | 0.0 % | 0 / 5 | 97.8 % | 44.5 % | 9.56 |
| mitigation | 19.6 % [0, 74.0] | 1 / 5 | 2.5 % | 2.0 % | 0.05 |

* More entropy mostly fixed the aligned exploration failure (4 / 5 seeds deliver, up from 2 / 5).
* **It did not change the weak-proxy result.** All 5 weak seeds still farmed and none delivered, even though
  delivering pays more. Under higher entropy the aligned and strong conditions are now separated on
  success, and the H2 rule is met on this post-hoc data.
* The mitigation went the other way (1 / 5 seeds delivering, down from 3 / 5). Its effect on success is not
  stable across settings, which agrees with treating the main run's H3 success comparison as unresolved.
  Its effect on farming (≤ 2.5 %) and hazard (2.0 %) was stable.

## 5. Discussion

**The incentive is not the whole story; the learning dynamics are.** The scripted rollouts show a reward
designer would be "safe" with a 0.03 checkpoint: gaming pays less than the task. PPO gamed it anyway, on
every seed, because it never found the task. A repeatable reward one step from the start is the easiest
improvement on doing nothing, and once found, a policy that shuttles there never collects the experience
that would show delivery is better. This is specification gaming in the sense that matters in practice: the
agent optimised what was easy to find in the reward, not what was optimal under it, and certainly not what
was intended. A proxy's danger depends on how *discoverable* it is relative to the true task, not only on
its size.

**Reported reward is a bad progress signal.** Across conditions, training return ranked policies in almost
exactly the reverse order of true success (figure 3). Someone monitoring only the training curve of the
strong condition would see a fast, stable rise to 9–10 and conclude training had worked.

**Gaming had a safety cost the reward never priced in.** The hazard penalty was the same in every
condition, yet farming agents entered the hazard in 44–57 % of episodes, against about 1 % for agents
pursuing the task. The proxy did not ask for risky behaviour. It made the agent spend its episode on a cell
next to the hazard, where slips do the rest. Side effects of gaming need to be measured separately from the
gaming itself.

**One-time payment removed the exploit.** Under mitigation, farming disappeared for both algorithms and hazard
rates returned to aligned levels. Whether it changes the chance of discovering the task could not be
determined at n = 5. The two algorithms point in opposite directions.

## 6. Limitations

* One hand-designed layout, one slip rate, one horizon. The results show that the mechanism *can* occur
  and how to measure it. They do not show how often it occurs.
* n = 5 seeds. Effects on the bimodal success metric are not resolvable at this sample size, and only
  near-deterministic effects (farming and hazard under proxies) are.
* All seeds share the same evaluation seeds (common random numbers). This helps comparisons between
  conditions but makes the seed-level interval collapse when seeds learn the same policy (section 3).
* One hyperparameter set per algorithm. DQN was not tuned, and the PPO exploration failure may disappear
  with other settings (section 4 tests one).
* The farming threshold (≥ 3 entries) is a modelling choice. Mean entry counts are reported alongside it.
* A 52-dimensional GridWorld says nothing direct about large models. Its value is as a clean, testable
  instance of the mechanism and of the evaluation practice: report the true objective, measure side effects,
  report seed counts for bimodal outcomes, and do not pool episodes.

## 7. Future work

* More seeds (20+) to resolve the bimodal success comparisons, especially aligned vs mitigation.
* Randomised start positions and layouts, to test whether gaming depends on the checkpoint being one step away.
* An exploration-bonus or curriculum baseline, to separate "the proxy is attractive" from "the task is hard to find".
* A proxy that is harder to discover than the task (checkpoint far from the start), to test the
  discoverability explanation directly.
* Potential-based shaping (Ng et al., 1999) as a principled alternative to the one-time mitigation.

## References

Amodei et al. (2016) *Concrete Problems in AI Safety*, arXiv:1606.06565 · Clark & Amodei (2016) *Faulty
Reward Functions in the Wild*, OpenAI · Krakovna et al. (2020) *Specification gaming: the flip side of AI
ingenuity*, DeepMind · Manheim & Garrabrant (2018) *Categorizing Variants of Goodhart's Law*,
arXiv:1803.04585 · Ng, Harada & Russell (1999) *Policy invariance under reward transformations*, ICML ·
Raffin et al. (2021) *Stable-Baselines3*, JMLR · Schulman et al. (2017) *Proximal Policy Optimization
Algorithms*, arXiv:1707.06347.
