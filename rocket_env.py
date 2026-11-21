import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import LunarLander

PAD_HALF_WIDTH = 0.2   # NEW: the pad spans x = -0.2 to +0.2 on Nova's dashboard
max_touchdown_speed=0.3
max_touchdown_angle=0.15
OUTCOME_BONUS = {           # NEW: reward v2, the judge's verdict becomes part of the reward
    "success":       +100,  # LunarLander already gives +100 for coming to rest: a real success totals +200
    "hard_landing":  -100,  # cancels LunarLander's +100: a hard landing is no longer "good"
    "off_pad":        -50,
    "tipped_over":    -50,  # on top of LunarLander's -100
    "timeout":       -100,  # hovering is a failure
    "crash":            0,  # already -100 from LunarLander
    "out_of_fuel":      0,
    "flew_away":        0,
}

class RocketLandingEnv(LunarLander):
    """LunarLander (continuous) with a fuel tank, a time limit and a landing judge."""

    def __init__(self, main_burn=0.004, side_burn=0.001, max_steps=1000,
                 max_touchdown_speed=0.3, max_touchdown_angle=0.15,
                 wind_range=None, reward_version=2, render_mode=None, **kwargs):
        if wind_range is not None:                 # NEW: random wind each flight
            kwargs["enable_wind"] = True
        super().__init__(continuous=True, render_mode=render_mode, **kwargs)
        self.wind_range = wind_range
        self.main_burn = main_burn
        self.side_burn = side_burn
        self.max_steps = max_steps                        # NEW: time limit
        self.max_touchdown_speed = max_touchdown_speed    # NEW: judge thresholds (to calibrate)
        self.max_touchdown_angle = max_touchdown_angle    # NEW: in radians (0.2 ≈ 11°)
        self.fuel = 1.0
        self.steps = 0                                    # NEW
        self.touchdown = None                             # NEW
        self.prev_obs = None              
        self.reward_version = reward_version       # NEW: 1 = LunarLander's reward, 2 = + judge bonus                # NEW

        low = np.append(self.observation_space.low, 0.0).astype(np.float32)
        high = np.append(self.observation_space.high, 1.0).astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.fuel = 1.0
        self.prev_obs = None                              # NEW
        obs, info = super().reset(seed=seed, options=options)
        if self.wind_range is not None:            # NEW: roll the dice for this flight's wind
            self.wind_power = float(self.np_random.uniform(*self.wind_range))
        self.steps = 0                                    # NEW: reset AFTER LunarLander's hidden step()
        self.touchdown = None                             # NEW
        self.prev_obs = obs                               # NEW
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.fuel <= 0.0:
            action = np.array([-1.0, 0.0], dtype=np.float32)

        m_power = (action[0] + 1.0) / 2.0 if action[0] > 0.0 else 0.0
        s_power = abs(action[1]) if abs(action[1]) > 0.5 else 0.0
        self.fuel = max(0.0, self.fuel - (m_power * self.main_burn + s_power * self.side_burn))

        obs, reward, terminated, truncated, info = super().step(action)
        obs = self._add_fuel(obs)

        # NEW: time limit
        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        # NEW: remember speed and angle just before the first leg touches
        legs_touching = obs[6] > 0 or obs[7] > 0
        if self.touchdown is None and self.prev_obs is not None and legs_touching:
            p = self.prev_obs
            self.touchdown = {
                "speed": float(np.hypot(p[2], p[3])),   # total speed = √(vx² + vy²)
                "angle": float(abs(p[4])),
                "x": float(p[0]),
            }
        self.prev_obs = obs

        info["fuel"] = self.fuel
        if terminated or truncated:   
            info["wind_power"] = self.wind_power if self.enable_wind else 0.0   # NEW: for analysis                    # NEW: the verdict
            info["outcome"] = self._judge(obs, truncated)
            if self.reward_version >= 2:           # NEW
                reward += OUTCOME_BONUS[info["outcome"]]
            info["touchdown"] = self.touchdown
        return obs, reward, terminated, truncated, info

    def _judge(self, obs, truncated):                     # NEW
        if truncated:
            return "timeout"
        if abs(obs[0]) >= 1.0:
            return "flew_away"
        if self.game_over:                                # the body hit the ground
            if self.fuel <= 0.0:
                return "out_of_fuel"
            if self.touchdown is None or self.touchdown["speed"] > self.max_touchdown_speed:
                return "crash"                            # came in too fast
            return "tipped_over"                          # touched down gently, then fell over
        # Otherwise the rocket came to rest
        if self.touchdown is None:
            return "crash"
        if abs(obs[0]) > PAD_HALF_WIDTH:
            return "off_pad"
        if obs[6] < 1 or obs[7] < 1:
            return "tipped_over"
        if (self.touchdown["speed"] > self.max_touchdown_speed
                or self.touchdown["angle"] > self.max_touchdown_angle):
            return "hard_landing"
        return "success"

    def _add_fuel(self, obs):
        return np.append(obs, self.fuel).astype(np.float32)