from collections import Counter
from rocket_env import RocketLandingEnv

def run(env, policy, n=50):
    outcomes = Counter()
    for i in range(n):
        obs, info = env.reset(seed=i)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(env))
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
    return dict(outcomes)

print("Engines off:      ", run(RocketLandingEnv(), lambda env: [-1.0, 0.0]))
print("Random pilot:     ", run(RocketLandingEnv(), lambda env: env.action_space.sample()))
print("Full throttle:    ", run(RocketLandingEnv(), lambda env: [1.0, 0.0]))
print("Tiny tank, gentle:", run(RocketLandingEnv(main_burn=0.05), lambda env: [0.3, 0.0]))