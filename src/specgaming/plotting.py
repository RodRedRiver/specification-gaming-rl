"""Research figures. Every error bar is a 95 % Student-t interval over independent training seeds."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .metrics import t_interval  # noqa: E402

COND_ORDER = ["aligned", "weak", "moderate", "strong", "mitigation"]
COND_COLORS = {"aligned": "#2b7bba", "weak": "#7fb3d5", "moderate": "#f0a032",
               "strong": "#c8402f", "mitigation": "#3a9d5d"}
ALGO_STYLE = {"ppo": dict(marker="o", ls="-"), "dqn": dict(marker="s", ls="--")}
ALGO_LABEL = {"ppo": "PPO", "dqn": "DQN"}
RATE_METRICS = {"success", "farming", "hazard"}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "legend.frameon": False})


def _algos(df) -> list[str]:
    present = list(dict.fromkeys(df["algo"]))
    return [a for a in ALGO_LABEL if a in present] + [a for a in present if a not in ALGO_LABEL]


def _err(row, m):
    mean = row[f"{m}_mean"]
    lo, hi = row[f"{m}_ci_low"], row[f"{m}_ci_high"]
    if np.isnan(lo):
        return mean, 0.0, 0.0
    if m in RATE_METRICS:  # clip rate intervals to [0, 1] for display
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    return mean, mean - lo, hi - mean


def _n_note(cond: pd.DataFrame) -> str:
    s = int(cond["n_seeds"].min())
    e = int(cond["n_episodes_per_seed"].min())
    return f"n = {s} training seeds per point, {e} evaluation episodes per seed; bars = 95% t-CI over seeds"


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def weight_curve(cond: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path, name: str, pct=True):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    scale = 100 if pct else 1
    for algo in _algos(cond):
        g = cond[cond["algo"] == algo]
        main = g[g["condition"] != "mitigation"].sort_values("proxy_weight")
        ys = [_err(r, metric) for _, r in main.iterrows()]
        ax.errorbar(main["proxy_weight"], [y[0] * scale for y in ys],
                    yerr=[[y[1] * scale for y in ys], [y[2] * scale for y in ys]],
                    capsize=3, label=f"{ALGO_LABEL.get(algo, algo)} (repeatable checkpoint)",
                    color="#333" if algo == "ppo" else "#888", **ALGO_STYLE.get(algo, {}))
        mit = g[g["condition"] == "mitigation"]
        for _, r in mit.iterrows():
            m, lo, hi = _err(r, metric)
            ax.errorbar([r["proxy_weight"] + 0.008], [m * scale], yerr=[[lo * scale], [hi * scale]], capsize=3,
                        color=COND_COLORS["mitigation"], marker=ALGO_STYLE.get(algo, {}).get("marker", "o"),
                        mfc="white", ls="none", label=f"{ALGO_LABEL.get(algo, algo)} mitigation (paid once)")
    ax.set_xlabel("Checkpoint proxy reward per entry (reward units)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if pct:
        ax.set_ylim(-3, 103)
    ax.legend(fontsize=7, loc="center right")
    fig.text(0.0, -0.06, _n_note(cond), fontsize=7, color="#555")
    _save(fig, out, name)


def reported_vs_true(seeds: pd.DataFrame, cond: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for algo in _algos(seeds):
        g = seeds[seeds["algo"] == algo]
        for c in COND_ORDER:
            s = g[g["condition"] == c]
            if s.empty:
                continue
            ax.scatter(s["return_train"], s["success"] * 100, color=COND_COLORS[c], alpha=0.55, s=18,
                       marker=ALGO_STYLE.get(algo, {}).get("marker", "o"), edgecolor="none")
            r = cond[(cond["algo"] == algo) & (cond["condition"] == c)].iloc[0]
            mx, lx, hx = _err(r, "return_train")
            my, ly, hy = _err(r, "success")
            ax.errorbar([mx], [my * 100], xerr=[[lx], [hx]], yerr=[[ly * 100], [hy * 100]], color=COND_COLORS[c],
                        marker=ALGO_STYLE.get(algo, {}).get("marker", "o"), ms=7, mec="black", capsize=2,
                        label=f"{ALGO_LABEL.get(algo, algo)} {c}")
    ax.set_xlabel("Reported (training) return per episode (reward units)")
    ax.set_ylabel("Genuine delivery success (% of episodes)")
    ax.set_title("Reported reward vs genuine task completion")
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=6.5, ncol=2, loc="best")
    fig.text(0.0, -0.06, "Small points: one training seed (mean of its evaluation episodes). Large points: "
             "condition mean with 95% t-CI over seeds.", fontsize=7, color="#555")
    _save(fig, out, "fig3_reported_vs_true")


def _bars(ax, cond, metric, algos, scale=1.0, conds=COND_ORDER):
    width = 0.8 / max(len(algos), 1)
    for i, algo in enumerate(algos):
        g = cond[cond["algo"] == algo].set_index("condition")
        present = [c for c in conds if c in g.index]
        x = np.array([conds.index(c) for c in present]) + (i - (len(algos) - 1) / 2) * width
        vals = [_err(g.loc[c], metric) for c in present]
        ax.bar(x, [v[0] * scale for v in vals], width * 0.92,
               yerr=[[v[1] * scale for v in vals], [v[2] * scale for v in vals]], capsize=2,
               color=[COND_COLORS[c] for c in present], edgecolor="black", linewidth=0.5,
               hatch=None if algo == "ppo" else "//", label=ALGO_LABEL.get(algo, algo))
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=0)


def exploitation(cond: pd.DataFrame, out: Path):
    algos = _algos(cond)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    _bars(axes[0], cond, "farming", algos, 100)
    axes[0].set_ylabel("Episodes with ≥3 checkpoint entries (%)")
    axes[0].set_title("Checkpoint farming rate")
    axes[0].set_ylim(0, 105)
    _bars(axes[1], cond, "checkpoint_entries", algos)
    axes[1].set_ylabel("Checkpoint entries per episode (count)")
    axes[1].set_title("Checkpoint exploitation frequency")
    if len(algos) > 1:
        axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="white", ec="black"),
                                plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch="//")],
                       labels=[ALGO_LABEL.get(a, a) for a in algos], fontsize=7)
    fig.text(0.0, -0.05, _n_note(cond), fontsize=7, color="#555")
    fig.tight_layout()
    _save(fig, out, "fig4_checkpoint_exploitation")


def mitigation(cond: pd.DataFrame, out: Path):
    algos = _algos(cond)
    sub = ["aligned", "strong", "mitigation"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    for ax, (m, lab, sc) in zip(axes, [("success", "Delivery success (%)", 100),
                                      ("farming", "Farming rate (%)", 100),
                                      ("hazard", "Hazard violations (%)", 100),
                                      ("return_train", "Training return (reward units)", 1)]):
        _bars(ax, cond, m, algos, sc, conds=sub)
        ax.set_title(lab, fontsize=9)
        if sc == 100:
            ax.set_ylim(0, 105)
    fig.suptitle("Mitigation: checkpoint reward 0.30 paid once vs every entry", fontsize=10)
    fig.text(0.0, -0.06, _n_note(cond) + ("; hatched = DQN" if "dqn" in algos else ""), fontsize=7, color="#555")
    fig.tight_layout()
    _save(fig, out, "fig5_mitigation")


def decomposition(cond: pd.DataFrame, out: Path):
    comps = [("comp_delivery", "delivery", "#3a9d5d"), ("comp_pickup", "pickup", "#8fc79a"),
             ("comp_checkpoint", "checkpoint (proxy)", "#f0a032"), ("comp_step", "step penalty", "#999999"),
             ("comp_hazard", "hazard penalty", "#c8402f")]
    algos = _algos(cond)
    fig, axes = plt.subplots(1, len(algos), figsize=(4.6 * len(algos), 3.2), squeeze=False)
    for ax, algo in zip(axes[0], algos):
        g = cond[cond["algo"] == algo].set_index("condition")
        present = [c for c in COND_ORDER if c in g.index]
        pos, neg = np.zeros(len(present)), np.zeros(len(present))
        for col, lab, color in comps:
            v = np.array([g.loc[c, f"{col}_mean"] for c in present])
            base = np.where(v >= 0, pos, neg)
            ax.bar(range(len(present)), v, bottom=base, color=color, label=lab, edgecolor="black", lw=0.4)
            pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present)
        ax.set_ylabel("Mean reward per episode (reward units)")
        ax.set_title(f"Reward decomposition ({ALGO_LABEL.get(algo, algo)})")
    axes[0][-1].legend(fontsize=7, loc="upper left")
    fig.text(0.0, -0.05, "Means over seeds of per-seed means; " + _n_note(cond).split(";")[0], fontsize=7, color="#555")
    fig.tight_layout()
    _save(fig, out, "fig7_reward_decomposition")


# ------------------------------------------------------------ learning curves
def load_monitor(run_dir: Path, n_envs: int) -> pd.DataFrame:
    frames = []
    for f in sorted(run_dir.glob("env*.monitor.csv")):
        d = pd.read_csv(f, skiprows=1)
        d["env_steps"] = d["l"].cumsum()
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames)
    d["timestep"] = d["env_steps"] * n_envs  # approximate global timestep (envs advance in lockstep)
    return d


def learning_curves(results: Path, cfg: dict, out: Path, n_bins: int = 30) -> pd.DataFrame:
    rows = []
    for algo in cfg["algorithms"]:
        hp = cfg["hyperparameters"][algo]
        total, n_envs = int(hp["total_timesteps"]), int(hp.get("n_envs", 1))
        edges = np.linspace(0, total, n_bins + 1)
        for c in cfg["conditions"]:
            for s in cfg["seeds"]:
                d = load_monitor(results / "train_logs" / f"{algo}__{c['name']}__seed{s}", n_envs)
                if d.empty:
                    continue
                d["bin"] = np.clip(np.digitize(d["timestep"], edges) - 1, 0, n_bins - 1)
                g = d.groupby("bin").agg(return_train=("r", "mean"), success=("is_success", "mean"),
                                         hazard=("hazard", "mean"), entries=("checkpoint_entries", "mean"))
                for b, r in g.iterrows():
                    rows.append({"algo": algo, "condition": c["name"], "seed": s,
                                 "timestep": (edges[b] + edges[b + 1]) / 2, **r.to_dict()})
    curves = pd.DataFrame(rows)
    if curves.empty:
        return curves
    for algo, ga in curves.groupby("algo", sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
        for c in COND_ORDER:
            gc = ga[ga["condition"] == c]
            if gc.empty:
                continue
            agg = gc.groupby("timestep")
            for ax, m, sc in zip(axes, ("return_train", "success", "hazard"), (1, 100, 100)):
                stats_ = agg[m].apply(lambda v: t_interval(v.values)).apply(pd.Series)
                stats_.columns = ["mean", "lo", "hi"]
                ax.plot(stats_.index, stats_["mean"] * sc, color=COND_COLORS[c], label=c, lw=1.3)
                ax.fill_between(stats_.index, stats_["lo"] * sc, stats_["hi"] * sc, color=COND_COLORS[c],
                                alpha=0.15, lw=0)
        axes[0].set_ylabel("Training return (reward units)")
        axes[1].set_ylabel("Delivery success (% of training episodes)")
        axes[2].set_ylabel("Hazard violations (% of training episodes)")
        for ax in axes:
            ax.set_xlabel("Training timesteps (thousands)")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
        axes[1].set_ylim(-3, 103)
        axes[2].set_ylim(-3, 103)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{ALGO_LABEL.get(algo, algo)} training curves (stochastic training policy, slips on)", fontsize=10)
        fig.text(0.0, -0.06, f"Mean over {len(cfg['seeds'])} seeds, band = 95% t-CI over seeds; "
                 f"{n_bins} bins of training timesteps.", fontsize=7, color="#555")
        fig.tight_layout()
        _save(fig, out, f"fig6_learning_curves_{algo}")
    return curves


# --------------------------------------------------------- trajectory figures
def visitation_heatmaps(results: Path, cfg: dict, algo: str, env_factory, out: Path):
    conds = [c["name"] for c in cfg["conditions"]]
    fig, axes = plt.subplots(1, len(conds), figsize=(2.3 * len(conds), 2.6))
    cmap = LinearSegmentedColormap.from_list("v", ["#ffffff", "#1f3b73"])
    for ax, c in zip(np.atleast_1d(axes), conds):
        env = env_factory(c)
        counts = np.zeros((env.n_rows, env.n_cols))
        n_eps = 0
        for s in cfg["seeds"]:
            p = results / "trajectories" / f"{algo}__{c}__seed{s}.json"
            if not p.exists():
                continue
            for traj in json.loads(p.read_text()):
                n_eps += 1
                for r, col in traj:
                    counts[r, col] += 1
        if n_eps:
            counts /= n_eps
        ax.imshow(counts, cmap=cmap, vmin=0, vmax=max(counts.max(), 1e-9))
        for (r, col) in env.hazards:
            ax.add_patch(plt.Rectangle((col - 0.5, r - 0.5), 1, 1, fill=False, ec="#c8402f", lw=1.5))
        for pos, lab in ((env.checkpoint, "C"), (env.package_pos, "P"), (env.destination, "D"), (env.start, "S")):
            ax.text(pos[1], pos[0], lab, ha="center", va="center", fontsize=8, color="#f0a032" if lab == "C" else "#333",
                    fontweight="bold")
        ax.set_title(f"{c}\n(max {counts.max():.1f} visits/episode)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"{ALGO_LABEL.get(algo, algo)}: mean visits per cell per evaluation episode (all seeds pooled); "
                 "red outline = hazard", fontsize=9)
    fig.tight_layout()
    _save(fig, out, f"fig8_visitation_{algo}")


def layout_diagram(env, out: Path):
    from .render import draw_grid

    env.reset(seed=0)
    img = draw_grid(env, cell=64)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("DeliveryGridWorld-v1 layout (illustrative)", fontsize=9)
    legend = [("hazard (episode ends, violation)", "#d64c3e"), ("checkpoint C (proxy reward)", "#f0be32"),
              ("package", "#785532"), ("destination D", "#46965a"), ("agent", "#285ac8"), ("start S", "#d7d7e6")]
    for i, (lab, col) in enumerate(legend):
        ax.add_patch(plt.Rectangle((img.shape[1] + 15, 20 + i * 45), 25, 25, color=col, clip_on=False))
        ax.text(img.shape[1] + 48, 33 + i * 45, lab, va="center", fontsize=7)
    _save(fig, out, "fig0_environment_layout")
