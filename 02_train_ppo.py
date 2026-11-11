from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 16 copies of the world running at the same time
env = make_vec_env("LunarLander-v3", n_envs=16, env_kwargs={"continuous": True}, seed=0)

# Nova (actor) + the Coach (critic), with PPO's settings
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,        # each rocket flies 1024 steps before a lesson
    batch_size=64,       # study the footage in groups of 64 moments
    n_epochs=4,          # re-study each round's footage 4 times
    gamma=0.999,         # how much future rewards matter
    gae_lambda=0.98,     # how the Coach blends real results with predictions
    ent_coef=0.01,       # small bonus for staying curious (exploration)
    verbose=1,           # print progress in the terminal
    tensorboard_log="runs/",  # save the curves for TensorBoard
    device="cpu",        # small networks run faster on CPU than on the Mac GPU
    seed=0,
)

model.learn(total_timesteps=1_000_000)   # 1 million steps of practice
model.save("models/ppo_lander")
print("Training done, model saved.")