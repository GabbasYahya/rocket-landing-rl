import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import LunarLander

PAD_HALF_WIDTH = 0.2          # the pad spans x = -0.2 to +0.2 on the dashboard
LANDER_DENSITY = 5.0          # PHASE 3: LunarLander's default body density

# PHASE 3: the "normal" world, used when a scenario doesn't specify a parameter
NOMINAL = {"gravity": -10.0, "mass_scale": 1.0, "wind": 0.0, "turbulence": 1.5, "fuel": 1.0}

OUTCOME_BONUS = {             # reward v2: the judge's verdict becomes part of the reward
    "success": +100, "hard_landing": -100, "off_pad": -50, "tipped_over": -50,
    "timeout": -100, "crash": 0, "out_of_fuel": 0, "flew_away": 0,
}


class RocketLandingEnv(LunarLander):
    """LunarLander (continuous) + fuel, time limit, landing judge, reward v2, randomised conditions."""

    def __init__(self, main_burn=0.004, side_burn=0.001, max_steps=1000,
                 max_touchdown_speed=0.3, max_touchdown_angle=0.15,
                 reward_version=2, ranges=None, wind_range=None,
                 render_mode=None, **kwargs):
        self.ranges = dict(ranges or {})                    # PHASE 3: {"gravity": (low, high), ...}
        if wind_range is not None:                          # keeps your Phase 2 scripts working
            self.ranges["wind"] = wind_range
        super().__init__(continuous=True, render_mode=render_mode, **kwargs)

        self.main_burn = main_burn
        self.side_burn = side_burn
        self.max_steps = max_steps
        self.max_touchdown_speed = max_touchdown_speed
        self.max_touchdown_angle = max_touchdown_angle
        self.reward_version = reward_version
        self.fuel = 1.0
        self.steps = 0
        self.touchdown = None
        self.prev_obs = None
        self.conditions = {}                                # PHASE 3: this flight's conditions

        low = np.append(self.observation_space.low, 0.0).astype(np.float32)
        high = np.append(self.observation_space.high, 1.0).astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.fuel = 1.0
        self.prev_obs = None
        obs, info = super().reset(seed=seed, options=options)

        # PHASE 3: choose this flight's conditions
        if options and "conditions" in options:             # exact scenario (stress test)
            self.conditions = {**NOMINAL, **options["conditions"]}
        elif self.ranges:                                   # random world (training)
            rolled = {k: float(self.np_random.uniform(lo, hi)) for k, (lo, hi) in self.ranges.items()}
            self.conditions = {**NOMINAL, **rolled}
        else:                                               # plain env: change nothing
            self.conditions = {}
        if self.conditions:
            self._apply_conditions()
            obs = self.step(np.array([-1.0, 0.0], dtype=np.float32))[0]   # fresh dashboard

        self.steps = 0
        self.touchdown = None
        self.prev_obs = obs
        return obs, info

    def _apply_conditions(self):                            # PHASE 3
        c = self.conditions
        self.world.gravity = (0.0, c["gravity"])
        for fixture in self.lander.fixtures:
            fixture.density = LANDER_DENSITY * c["mass_scale"]
        self.lander.ResetMassData()
        self.enable_wind = c["wind"] > 0.0
        self.wind_power = c["wind"]
        self.turbulence_power = c["turbulence"]
        self.wind_idx = int(self.np_random.integers(-9999, 9999))     # random gust pattern
        self.torque_idx = int(self.np_random.integers(-9999, 9999))
        self.fuel = c["fuel"]

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.fuel <= 0.0:
            action = np.array([-1.0, 0.0], dtype=np.float32)

        m_power = (action[0] + 1.0) / 2.0 if action[0] > 0.0 else 0.0
        s_power = abs(action[1]) if abs(action[1]) > 0.5 else 0.0
        self.fuel = max(0.0, self.fuel - (m_power * self.main_burn + s_power * self.side_burn))

        obs, reward, terminated, truncated, info = super().step(action)
        obs = self._add_fuel(obs)

        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        legs_touching = obs[6] > 0 or obs[7] > 0
        if self.touchdown is None and self.prev_obs is not None and legs_touching:
            p = self.prev_obs
            self.touchdown = {"speed": float(np.hypot(p[2], p[3])),
                              "angle": float(abs(p[4])), "x": float(p[0])}
        self.prev_obs = obs

        info["fuel"] = self.fuel
        if terminated or truncated:
            info["outcome"] = self._judge(obs, truncated)
            info["touchdown"] = self.touchdown
            info["wind_power"] = self.wind_power if self.enable_wind else 0.0
            info["conditions"] = dict(self.conditions)      # PHASE 3: for analysis
            if self.reward_version >= 2:
                reward += OUTCOME_BONUS[info["outcome"]]
        return obs, reward, terminated, truncated, info

    def _judge(self, obs, truncated):
        if truncated:
            return "timeout"
        if abs(obs[0]) >= 1.0:
            return "flew_away"
        if self.game_over:
            if self.fuel <= 0.0:
                return "out_of_fuel"
            if self.touchdown is None or self.touchdown["speed"] > self.max_touchdown_speed:
                return "crash"
            return "tipped_over"
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