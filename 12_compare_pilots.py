from collections import Counter
import numpy as np
from stable_baselines3 import PPO
from rocket_env import RocketLandingEnv



pilots = {
    "no-wind Nova": PPO.load("models/ppo_fuel", device="cpu"),
    "wind Nova   ": PPO.load("models/ppo_wind", device="cpu"),
    "wind Nova v2": PPO.load("models/ppo_wind_v2", device="cpu"),
}

for wind in [0, 10, 15]:
    print(f"\n=== wind {wind} ===")
    for name, model in pilots.items():
        env = RocketLandingEnv(enable_wind=wind > 0, wind_power=float(wind))
        outcomes, scores, speeds, angles = Counter(), [], [], []
        for i in range(100):
            obs, info = env.reset(seed=1000 + i)
            done, total = False, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                done = terminated or truncated
            outcomes[info["outcome"]] += 1
            scores.append(total)
            if info["touchdown"] is not None:
                speeds.append(info["touchdown"]["speed"])
                angles.append(info["touchdown"]["angle"])
        env.close()
        print(f"{name} | success {outcomes['success']:3d}% | score {np.mean(scores):6.1f} | "
              f"touchdown speed {np.median(speeds):.3f} | angle {np.median(angles):.3f}")