
import json
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from mindscheduler.study_env import StudyEnv
from mindscheduler.baselines import equal_split

SEEDS = [0, 1]
N_EPISODES = 100
EVAL_SEED0 = 1000          


def play(env, seed, model=None, baseline=None):
    obs, _ = env.reset(seed=seed)
    state, start, done = None, np.array([True]), False
    while not done:
        if baseline is not None:
            action = baseline(env)
        else:
            action, state = model.predict(obs, state=state, episode_start=start,
                                          deterministic=False)
            action = int(action)
            start = np.array([False])
        obs, _, done, _, info = env.step(action)
    return float(info["marks"].mean())


def load(algo, seed):
    cls = PPO if algo == "ppo" else RecurrentPPO
    return cls.load(f"results/models/{algo}_seed{seed}")


if __name__ == "__main__":
    results = {}
    env = StudyEnv()
    for algo in ["ppo", "rppo"]:
        per_seed, all_scores = [], []
        for s in SEEDS:
            model = load(algo, s)
            scores = [play(env, EVAL_SEED0 + e, model=model) for e in range(N_EPISODES)]
            per_seed.append(float(np.mean(scores)))
            all_scores += scores
            print(f"{algo} seed {s}: {np.mean(scores):.2f}")
        results[algo] = {"per_seed_mean": per_seed, "episode_scores": all_scores}
    with open("results/agent_results.json", "w") as f:
        json.dump(results, f)

    noise_results = {}
    for noise in [0.05, 0.10, 0.20, 0.30]:
        nenv = StudyEnv(obs_noise=noise)
        row = {"equal_split": float(np.mean(
            [play(nenv, EVAL_SEED0 + e, baseline=equal_split) for e in range(50)]))}
        for algo in ["ppo", "rppo"]:
            vals = []
            for s in SEEDS:
                model = load(algo, s)
                vals += [play(nenv, EVAL_SEED0 + e, model=model) for e in range(50)]
            row[algo] = float(np.mean(vals))
        noise_results[str(noise)] = row
        print(noise, row)
    with open("results/noise_results.json", "w") as f:
        json.dump(noise_results, f, indent=2)
