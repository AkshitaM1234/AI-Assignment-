import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StudyEnv(gym.Env):
    LEARN, REVISE, TEST = 0, 1, 2

    def __init__(self, n_subjects=3, topics_per_subject=4, n_days=30,
                 slots_per_day=4, exam_days=(20, 25, 30), obs_noise=0.10,
                 shape_coef=2.0, exam_weight=10.0, burnout_penalty=1.0):
        super().__init__()
        assert len(exam_days) == n_subjects and max(exam_days) <= n_days
        self.S, self.T = n_subjects, topics_per_subject
        self.N = self.S * self.T
        self.n_days, self.slots_per_day = n_days, slots_per_day
        self.exam_days = np.array(exam_days)
        self.obs_noise = obs_noise
        self.shape_coef = shape_coef
        self.exam_weight = exam_weight
        self.burnout_penalty = burnout_penalty
        self.subject_of = np.repeat(np.arange(self.S), self.T)

        # actions: [0,N) learn topic i | [N,2N) revise | [2N,3N) practice test | 3N rest
        self.rest_action = 3 * self.N
        self.action_space = spaces.Discrete(3 * self.N + 1)
        obs_dim = 3 * self.N + 2 * self.S + 2
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

    # ---------- helpers ----------
    def encode(self, kind, topic):
        return kind * self.N + topic

    def decode(self, action):
        return action // self.N, action % self.N

    def active_topics(self):
        return np.where(~self.exam_done[self.subject_of])[0]

    def _gate(self, topic):
        if topic % self.T == 0:
            return 1.0
        return 0.2 + 0.8 * min(1.0, self.mastery[topic - 1] / 0.5)

    def _obs(self):
        n = self.n_days
        days_to = np.maximum(0, self.exam_days - self.day) / n
        obs = np.concatenate([
            self.est,                                  
            np.minimum(self.obs_age, n) / n,          
            np.minimum(self.since_study, n) / n,      
            days_to,                                  
            self.exam_done.astype(float),            
            [self.slot / self.slots_per_day, min(self.fatigue, 1.0)],
        ])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.eta = rng.uniform(0.15, 0.35)         
        self.lam = rng.uniform(0.02, 0.06)         
        self.difficulty = rng.uniform(0.6, 1.2, size=self.N)
        self.mastery = rng.uniform(0.0, 0.2, size=self.N)  
        self.strength = np.ones(self.N)                     
        self.est = np.zeros(self.N)
        self.obs_age = np.full(self.N, float(self.n_days))
        self.since_study = np.full(self.N, float(self.n_days))
        self.study_count = np.zeros(self.N, dtype=int)
        self.exam_done = np.zeros(self.S, dtype=bool)
        self.marks = np.zeros(self.S)
        self.fatigue, self.day, self.slot, self.t, self.burnouts = 0.0, 0, 0, 0, 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        phi_before = self.mastery.mean()
        reward, burnout = 0.0, False
        energy = max(0.05, 1.0 - self.fatigue ** 2)

        if action == self.rest_action:
            self.fatigue = max(0.0, self.fatigue - 0.4)
        else:
            kind, topic = self.decode(action)
            if self.exam_done[self.subject_of[topic]]:
                self.fatigue += 0.10                    
            else:
                m = self.mastery[topic]
                base = self.eta * self.difficulty[topic] * (1 - m) * energy * self._gate(topic)
                if kind == self.LEARN:
                    self.mastery[topic] += base
                    self.strength[topic] += 0.1
                    self.fatigue += 0.15
                elif kind == self.REVISE:
                    self.mastery[topic] += 0.4 * base
                    self.strength[topic] += 0.5
                    self.fatigue += 0.12
                else:                                     
                    noisy = m + self.np_random.normal(0.0, self.obs_noise)
                    self.est[topic] = float(np.clip(noisy, 0.0, 1.0))
                    self.obs_age[topic] = 0.0
                    self.mastery[topic] += 0.05 * (1 - m)
                    self.strength[topic] += 0.1
                    self.fatigue += 0.05
                self.since_study[topic] = 0.0
                self.study_count[topic] += 1

        if self.fatigue >= 1.0:                         
            burnout = True
            self.burnouts += 1
            reward -= self.burnout_penalty
            self.fatigue = 0.6
            self.slot = self.slots_per_day
        else:
            self.slot += 1
        self.t += 1

        if self.slot >= self.slots_per_day:
            reward += self._end_of_day()

        reward += self.shape_coef * (self.mastery.mean() - phi_before)
        terminated = bool(self.exam_done.all())
        info = {"marks": self.marks.copy(), "burnout": burnout}
        return self._obs(), float(reward), terminated, False, info

    def _end_of_day(self):
        self.day += 1
        self.slot = 0
        reward = 0.0
        for k in range(self.S):                            
            if not self.exam_done[k] and self.day == self.exam_days[k]:
                sl = slice(k * self.T, (k + 1) * self.T)
                marks = 100 * self.mastery[sl].mean() + self.np_random.normal(0.0, 3.0)
                self.marks[k] = float(np.clip(marks, 0.0, 100.0))
                self.exam_done[k] = True
                reward += self.exam_weight * self.marks[k] / 100.0
        active = ~self.exam_done[self.subject_of]         
        self.mastery[active] *= np.exp(-self.lam / self.strength[active])
        self.fatigue = max(0.0, self.fatigue - 0.5)      
        self.since_study += 1
        self.obs_age += 1
        return reward
