import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

model = PPO.load("models/ppo_lander", device="cpu")

env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", name_prefix="phase1_ppo",
                  episode_trigger=lambda episode_id: True)   # record every episode

for i in range(3):
    obs, info = env.reset(seed=200 + i)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
env.close()
print("Videos saved in the videos/ folder")