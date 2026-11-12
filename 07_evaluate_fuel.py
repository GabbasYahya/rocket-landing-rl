import numpy as np
from stable_baselines3 import PPO
from rocket_env import RocketLandingEnv

model = PPO.load("models/ppo_fuel", device="cpu")
env = RocketLandingEnv()

scores, fuel_left, ran_out = [], [], 0
for i in range(100):
    obs, info = env.reset(seed=1000 + i)
    done, total = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    scores.append(total)
    fuel_left.append(info["fuel"])
    if info["fuel"] <= 0.0:
        ran_out += 1
env.close()

scores, fuel_left = np.array(scores), np.array(fuel_left)
print(f"Average score:      {scores.mean():.1f} ± {scores.std():.1f}")
print(f"Success (≥ 200):    {(scores >= 200).mean() * 100:.0f}%")
print(f"Average fuel left:  {fuel_left.mean() * 100:.0f}% of the tank")
print(f"Ran out of fuel:    {ran_out}%")