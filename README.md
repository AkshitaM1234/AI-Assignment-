# MindScheduler
Reinforcement learning for semester study planning under uncertainty.

A simulated student (3 subjects x 4 topics, 30 days) has hidden topic mastery and sees
only noisy practice-test results. Each study slot the agent chooses to learn, revise,
test a topic, or rest, to maximise final exam marks. Includes forgetting, prerequisites,
fatigue/burnout and staggered exams.

## Setup
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

## Reproduce
    python -m pytest -q
    python run_baselines.py
    python train.py --algo ppo --seed 0 --steps 100000
    python train.py --algo ppo --seed 1 --steps 100000
    python train.py --algo rppo --seed 0 --steps 100000
    python train.py --algo rppo --seed 1 --steps 100000
    python evaluate_agents.py
    python plots.py

## Structure
- mindscheduler/study_env.py: custom Gymnasium environment
- mindscheduler/baselines.py: rule-based baselines
- train.py, evaluate_agents.py, plots.py: training, evaluation, figures
- results/: trained models, logs, JSON results, figures