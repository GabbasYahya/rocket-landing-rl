import gymnasium as gym

env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
obs, info = env.reset(seed=0)

print("Observation (Nova's dashboard):", obs)
print("Action space (Nova's sliders):", env.action_space)

total_reward = 0
done = False
while not done:
    action = env.action_space.sample()          # random sliders: Nova knows nothing yet
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print("Episode finished. Total reward:", round(total_reward, 1))
env.close()