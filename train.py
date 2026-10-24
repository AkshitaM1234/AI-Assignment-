import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from mindscheduler.study_env import StudyEnv

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["ppo", "rppo"], required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=300_000)
args = parser.parse_args()

os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

# Monitor writes per-episode reward/length to results/logs/<name>.monitor.csv
env = Monitor(StudyEnv(), f"results/logs/{args.algo}_seed{args.seed}")

common = dict(learning_rate=3e-4, n_steps=1024, gamma=0.99, ent_coef=0.01,
              seed=args.seed, verbose=0)
if args.algo == "ppo":
    model = PPO("MlpPolicy", env, batch_size=128, **common)
else:
    model = RecurrentPPO("MlpLstmPolicy", env,
                         policy_kwargs=dict(lstm_hidden_size=64), **common)

model.learn(total_timesteps=args.steps, progress_bar=False)
model.save(f"results/models/{args.algo}_seed{args.seed}")
print(f"saved results/models/{args.algo}_seed{args.seed}.zip")
