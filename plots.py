import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)
LABELS = {"ppo": "PPO", "rppo": "RecurrentPPO"}
COLORS = {"ppo": "tab:blue", "rppo": "tab:red"}


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fig_training():
    plt.figure(figsize=(3.4, 2.5))
    for algo in ["ppo", "rppo"]:
        for i, f in enumerate(sorted(glob.glob(f"results/logs/{algo}_seed*.monitor.csv"))):
            data = np.atleast_2d(np.loadtxt(f, delimiter=",", skiprows=2))
            r, steps = data[:, 0], np.cumsum(data[:, 1])
            w = min(50, len(r))
            plt.plot(steps[w - 1:], smooth(r, w), color=COLORS[algo], alpha=0.7,
                     label=LABELS[algo] if i == 0 else None)
    plt.xlabel("Training steps")
    plt.ylabel("Episode reward (smoothed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/fig_training.pdf")
    plt.close()


def fig_results():
    base = json.load(open("results/baseline_results.json"))
    agents = json.load(open("results/agent_results.json"))
    names, means, errs, colors = [], [], [], []
    for k, v in base.items():
        names.append(k.replace("_", " "))
        means.append(v["mean_marks"]); errs.append(v["std"]); colors.append("tab:gray")
    for algo in ["ppo", "rppo"]:
        sc = np.array(agents[algo]["episode_scores"])
        names.append(LABELS[algo]); means.append(sc.mean()); errs.append(sc.std())
        colors.append(COLORS[algo])
    plt.figure(figsize=(3.4, 2.8))
    plt.barh(names, means, xerr=errs, color=colors, capsize=2)
    plt.xlabel("Mean exam marks (error bar = 1 std over episodes)")
    plt.tight_layout()
    plt.savefig("results/figures/fig_results.pdf")
    plt.close()


def fig_noise():
    nr = json.load(open("results/noise_results.json"))
    noises = sorted(nr.keys(), key=float)
    plt.figure(figsize=(3.4, 2.5))
    for key, label, color in [("equal_split", "Equal split", "tab:gray"),
                              ("ppo", "PPO", "tab:blue"),
                              ("rppo", "RecurrentPPO", "tab:red")]:
        plt.plot([float(n) for n in noises], [nr[n][key] for n in noises],
                 marker="o", label=label, color=color)
    plt.xlabel("Observation noise (std)")
    plt.ylabel("Mean exam marks")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/fig_noise.pdf")
    plt.close()


if __name__ == "__main__":
    fig_training()
    fig_results()
    fig_noise()
    print("figures saved in results/figures/")
