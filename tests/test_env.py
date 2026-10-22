import numpy as np
import pytest
from mindscheduler.study_env import StudyEnv


def test_reset_and_obs_shape():
    env = StudyEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_episode_terminates_and_stays_bounded():
    env = StudyEnv()
    env.reset(seed=1)
    done, steps = False, 0
    while not done:
        obs, r, done, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.all(env.mastery >= 0) and np.all(env.mastery <= 1)
        steps += 1
        assert steps <= env.n_days * env.slots_per_day
    assert env.exam_done.all()


def test_prerequisite_gating():
    env = StudyEnv()
    env.reset(seed=2)
    env.mastery[:] = 0.0
    assert env._gate(1) == pytest.approx(0.2)
    env.mastery[0] = 0.5
    assert env._gate(1) == pytest.approx(1.0)
    assert env._gate(0) == 1.0


def test_forgetting_reduces_mastery():
    env = StudyEnv()
    env.reset(seed=3)
    env.mastery[:] = 0.8
    before = env.mastery.copy()
    for _ in range(env.slots_per_day):
        env.step(env.rest_action)
    assert np.all(env.mastery < before)


def test_same_seed_same_episode():
    a, b = StudyEnv(), StudyEnv()
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert np.allclose(oa, ob)
    assert np.allclose(a.step(5)[0], b.step(5)[0])
