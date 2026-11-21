from collections import Counter
from stable_baselines3 import PPO
from rocket_env import RocketLandingEnv

model = PPO.load("models/ppo_wind", device="cpu")   # Nova, trained WITH random wind 0–15

for wind in [0, 5, 10, 15, 20]:
    env = RocketLandingEnv(enable_wind=wind > 0, wind_power=float(wind))
    outcomes = Counter()
    for i in range(100):
        obs, info = env.reset(seed=1000 + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
    env.close()
    print(f"wind {wind:2d}: success {outcomes['success']:3d}% | {dict(outcomes)}")