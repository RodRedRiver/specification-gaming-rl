## Fellowship description

*Every number below is taken from the tables in this file (main run) and `results_followup_entropy.md` (post-hoc follow-up).*

I built a reproducible testbed for specification gaming: a delivery GridWorld whose reward adds a checkpoint bonus paid on every entry, next to a hazard the reward never mentions. Across 50 preregistered PPO and DQN runs (5 seeds per condition), every PPO agent given a repeatable checkpoint learned the same exploit, shuttling in and out of it. Delivery success fell to 0% while training return rose from −0.10 to 9.65, and 43.6% of episodes ended in the hazard, against 1.1% for aligned agents. This happened even at a weight where scripted rollouts show gaming pays less than the task, contradicting my preregistered hypothesis. The logs trace it to the proxy being easier to discover than the task, and a post-hoc follow-up with more exploration reproduced it on 5 of 5 seeds. Paying the checkpoint only once removed the exploit (farming ≤ 0.2% in the main run).

## CV bullets

* Designed and ran a preregistered reinforcement-learning study of specification gaming: a custom Gymnasium environment, PPO and DQN (Stable-Baselines3), 75 training runs over 5 seeds per condition, seed-level confidence intervals and 27 automated tests
* Showed that a repeatable proxy reward captured 15 of 15 PPO runs (0% task success while training return rose from −0.10 to 9.65) and raised safety violations from 1.1% to 43.6% of episodes, including at a reward weight where gaming paid less than the task
* Reported a contradicted preregistered hypothesis, traced it to exploration and tested that in a labelled post-hoc follow-up (capture persisted on 5 of 5 seeds); showed that paying the reward once cut farming to ≤ 0.2% and violations to 1.5–1.8% (PPO, DQN)
