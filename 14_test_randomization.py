from stable_baselines3.common.env_checker import check_env
from rocket_env import RocketLandingEnv

TRAIN_RANGES = {
    "gravity": (-12.0, -8.0),
    "mass_scale": (0.8, 1.2),
    "wind": (0.0, 15.0),
    "turbulence": (0.0, 1.5),
    "fuel": (0.7, 1.0),
}

env = RocketLandingEnv(ranges=TRAIN_RANGES)
check_env(env)
print("check_env passed\n")

# 1. Every reset should roll a different world
for i in range(3):
    obs, info = env.reset(seed=i)
    print({k: round(v, 2) for k, v in env.conditions.items()}, "| fuel on dashboard:", round(float(obs[-1]), 2))

# 2. Does gravity really change? Free fall with engines off
def fall_steps(g):
    obs, info = env.reset(seed=0, options={"conditions": {"gravity": g}})
    for step in range(1000):
        obs, reward, terminated, truncated, info = env.step([-1.0, 0.0])
        if terminated or truncated:
            return step

print("\nSteps to hit the ground, gravity -6: ", fall_steps(-6.0))
print("Steps to hit the ground, gravity -12:", fall_steps(-12.0))

# 3. Does mass really change?
for m in [0.8, 1.0, 1.2]:
    env.reset(seed=0, options={"conditions": {"mass_scale": m}})
    print(f"mass_scale {m}: rocket mass = {env.lander.mass:.2f}")
env.close()