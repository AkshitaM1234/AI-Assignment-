import json
import numpy as np
from mindscheduler.study_env import StudyEnv
from mindscheduler.baselines import POLICIES


def evaluate(policy, n_episodes=200, seed0=1000):
    env = StudyEnv()
    scores, burnouts = [], []
    for ep in range(n_episodes):
        env.reset(seed=seed0 + ep)
        done = False
        while not done:
            _, _, done, _, info = env.step(policy(env))
        scores.append(info["marks"].mean())
        burnouts.append(env.burnouts)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(burnouts))


if __name__ == "__main__":
    results = {}
    print(f"{'policy':24s} {'mean marks':>10s} {'std':>7s} {'burnouts':>9s}")
    for name, pol in POLICIES.items():
        mean, std, burn = evaluate(pol)
        results[name] = {"mean_marks": mean, "std": std, "burnouts": burn}
        print(f"{name:24s} {mean:10.2f} {std:7.2f} {burn:9.2f}")
    with open("results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
