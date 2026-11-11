import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("models/ppo_lander", device="cpu")    # load Nova's trained brain

# --- Part 1: watch 3 flights ---
env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
for i in range(3):
    obs, info = env.reset(seed=100 + i)
    done, total = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)   # her best move, no noise
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    print(f"Watched flight {i + 1}: {total:.1f}")
env.close()

# --- Part 2: the exam, 100 flights without a window (fast) ---
env = gym.make("LunarLander-v3", continuous=True)
scores = []
for i in range(100):
    obs, info = env.reset(seed=1000 + i)     # seeds she never trained on
    done, total = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    scores.append(total)
env.close()

scores = np.array(scores)
print(f"\nAverage score: {scores.mean():.1f} ± {scores.std():.1f}")
print(f"Success (score ≥ 200): {(scores >= 200).mean() * 100:.0f}%")
print(f"Bad flights (score < 0): {(scores < 0).mean() * 100:.0f}%")