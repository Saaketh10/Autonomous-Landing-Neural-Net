import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces
import numpy as np

# Building the environment
environment_name = 'LunarLander-v2'
env = gym.make(environment_name, render_mode="rgb_array")
obs = env.reset()

# Building the SAC model
# Assuming the original action space is Discrete(4)
# You need to choose an appropriate continuous action space for SAC
# Example: Box action space with 2 dimensions
# Adjust the low and high values based on the requirements of your environment
env.action_space = spaces.Box([-1.5 -1.5 -5. -5. -3.1415927 -5. -0. -0. ], [1.5 1.5 5. 5. 3.1415927 5. 1. 1. ], (8,), float32)
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
# Testing the SAC model
evaluate_policy(model, env, n_eval_episodes=10, render=True)

# Saving and reloading the SAC model
model.save("SAC_model")
del model
model = SAC.load("SAC_model", env=env)