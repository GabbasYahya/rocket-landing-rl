from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rocket_env import RocketLandingEnv

env = make_vec_env(RocketLandingEnv, n_envs=16, seed=0)   # your class instead of "LunarLander-v3"

model = PPO(
    "MlpPolicy", env,
    n_steps=1024,
    batch_size=64, 
    n_epochs=4,
    gamma=0.999, 
    gae_lambda=0.98, 
    ent_coef=0.01,
    verbose=1, 
    tensorboard_log="runs/", 
    device="cpu", 
    seed=0,
)
model.learn(total_timesteps=1_000_000, tb_log_name="ppo_fuel")
model.save("models/ppo_fuel")
print("Training done, model saved.")