import numpy as np

REST_FATIGUE = 0.75


def random_policy(env):
    return env.action_space.sample()


def equal_split(env):
    if env.fatigue > REST_FATIGUE:
        return env.rest_action
    act = env.active_topics()
    return env.encode(env.LEARN, act[env.t % len(act)])


def earliest_exam_first(env):
    if env.fatigue > REST_FATIGUE:
        return env.rest_action
    pending = [k for k in range(env.S) if not env.exam_done[k]]
    k = min(pending, key=lambda s: env.exam_days[s])
    topics = np.arange(k * env.T, (k + 1) * env.T)
    i = topics[np.argmin(env.study_count[topics])]
    kind = env.LEARN if env.study_count[i] < 3 else env.REVISE
    return env.encode(kind, i)


def spaced_rotation(env):
    if env.fatigue > REST_FATIGUE:
        return env.rest_action
    act = env.active_topics()
    i = act[np.argmax(env.since_study[act])]
    kind = env.LEARN if env.study_count[i] < 3 else env.REVISE
    return env.encode(kind, i)


def weakest_first_oracle(env):
    if env.fatigue > REST_FATIGUE:
        return env.rest_action
    act = env.active_topics()
    return env.encode(env.LEARN, act[np.argmin(env.mastery[act])])


POLICIES = {
    "random": random_policy,
    "equal_split": equal_split,
    "earliest_exam_first": earliest_exam_first,
    "spaced_rotation": spaced_rotation,
    "weakest_first_oracle": weakest_first_oracle,
}
