import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import LunarLander


class RocketLandingEnv(LunarLander):
    """LunarLander (continuous) with a fuel tank."""

    def __init__(self, main_burn=0.004, side_burn=0.001, render_mode=None, **kwargs):
        super().__init__(continuous=True, render_mode=render_mode, **kwargs)
        self.main_burn = main_burn      # fuel used per step at full main throttle
        self.side_burn = side_burn      # fuel used per step at full side thrust
        self.fuel = 1.0                 # 1.0 = full tank, 0.0 = empty

        # Nova's dashboard grows from 8 to 9 numbers: the last one is fuel (0 to 1)
        low = np.append(self.observation_space.low, 0.0).astype(np.float32)
        high = np.append(self.observation_space.high, 1.0).astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.fuel = 1.0                 # refuel BEFORE LunarLander builds the first dashboard
        obs, info = super().reset(seed=seed, options=options)
        return obs, info                # already has fuel: LunarLander's reset called our step()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Empty tank: force both engines off
        if self.fuel <= 0.0:
            action = np.array([-1.0, 0.0], dtype=np.float32)

        # Same throttle rules as LunarLander (main: 0 or 0.5-1.0, side: 0 or 0.5-1.0)
        m_power = (action[0] + 1.0) / 2.0 if action[0] > 0.0 else 0.0
        s_power = abs(action[1]) if abs(action[1]) > 0.5 else 0.0

        # Burn fuel
        self.fuel = max(0.0, self.fuel - (m_power * self.main_burn + s_power * self.side_burn))

        # Let LunarLander do the physics, reward and termination
        obs, reward, terminated, truncated, info = super().step(action)
        info["fuel"] = self.fuel
        return self._add_fuel(obs), reward, terminated, truncated, info

    def _add_fuel(self, obs):
        return np.append(obs, self.fuel).astype(np.float32)