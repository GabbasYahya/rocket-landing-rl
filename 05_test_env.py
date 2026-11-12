from stable_baselines3.common.env_checker import check_env
from rocket_env import RocketLandingEnv

# 1. Automatic check: correct shapes, types, spaces?
env = RocketLandingEnv()
check_env(env)
print("check_env passed")
# 2. Tiny tank (empties after ~50 steps): does the engine really stop?
env = RocketLandingEnv(main_burn=0.004)
obs, info = env.reset(seed=0)
print("Dashboard size:", len(obs), "| starting fuel:", obs[-1])

for step in range(1000):
    obs, reward, terminated, truncated, info = env.step([1.0, 0.0])   # main engine at 100%
    if step % 10 == 0:
        print(f"step {step:4d} | fuel {info['fuel']:.3f} | x {obs[0]:+.2f} | "
              f"height {obs[1]:.2f} | vertical speed {obs[3]:+.2f}")
    if terminated or truncated:
        print(f"Episode ended at step {step}, fuel left: {info['fuel']:.3f}, reward {reward:.0f}")
        break
env.close()