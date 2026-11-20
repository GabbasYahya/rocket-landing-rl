from collections import Counter
import numpy as np
from stable_baselines3 import PPO
from rocket_env import RocketLandingEnv

model = PPO.load("models/ppo_fuel", device="cpu")
env = RocketLandingEnv()

outcomes, speeds, angles = Counter(), [], []
for i in range(100):
    obs, info = env.reset(seed=1000 + i)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    outcomes[info["outcome"]] += 1
    if info["touchdown"] is not None:
        speeds.append(info["touchdown"]["speed"])
        angles.append(info["touchdown"]["angle"])
env.close()

print("Outcomes:", dict(outcomes))
print(f"Touchdown speed: median {np.median(speeds):.3f} | 95% of flights below {np.percentile(speeds, 95):.3f} | max {np.max(speeds):.3f}")
print(f"Touchdown angle: median {np.median(angles):.3f} | 95% of flights below {np.percentile(angles, 95):.3f} | max {np.max(angles):.3f}")