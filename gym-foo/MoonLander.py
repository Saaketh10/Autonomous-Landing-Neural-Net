import gym
import pygame
import numpy as np
import gym_foo 
import cv2
#import gymnasium as gym
from stable_baselines3 import DQN 
#from stable_baselines.common.vec_env import DummyVecEnv #Dummy Vectorized Environment, some algorithms require a vectorized environment, so this works
from stable_baselines3.common.evaluation import evaluate_policy #Function used to evauluate performance of agent. Calculates Mean and Standard Deviation of all rewards obtained
import matplotlib.pyplot as plt

#BUILDING MODEL
environment_name = 'MoonLanding-v0'
env = gym.make(environment_name) #Creates Environment
#env = DummyVecEnv([lambda: env]) #Wraps the Environment in the Dummy Vectorized Environment
obs = env.reset()
model = DQN('MlpPolicy', env,verbose = 1) # Creates an ACER model with a Multi-Layer Perceptron (MLP) policy 
#When Verbose=1, it prints training info of each episode
model.learn(total_timesteps=100000) # Trains the model for a specified number of timesteps


#TESTING MODEL
reward,rewardstd = evaluate_policy(model, env, n_eval_episodes=10, render=True) #Evaluates 10 episodes and renders the trained model


#SAVING MODEL 
env.close()
model.save("DQN_model")
del model
model = DQN.load("DQN_model", env=env) #Reloading Model to restart
    #We Want High Variance and High Possible Mean Reward
vec_env = model.get_env()
obs = vec_env.reset()
rewards = np.zeros(5000)
actions = np.zeros(5000)
dones = False
i=1
while not dones:
    actions[i], _states = model.predict(obs, deterministic=True)
    obs, rewards[i], dones, info = vec_env.step(actions[i])
    i+=1
    if dones == True:
        break

action_to_direction = {
            0: np.array([0, 1]),    # Move right
            1: np.array([-1, 0]),   # Move up
            2: np.array([0, -1]),   # Move left
            3: np.array([1, 0]),    # Move down
            4: np.array([0, 0])     # Terminate
        }


position = init_position
# Plot image
filepath = 'gym-foo/Idem4.jpg'
gray_image = cv2.imread(filepath,0)

        
for j in [1:i]:
    action = actions[j]
    direction = action_to_direction[action]
    position = position + direction
    # Scatter plot position




New_vals = rewards[0:i]
plt.plot(New_vals)
plt.xlabel("Iterations")
plt.ylabel("Reward")
plt.savefig("MoonLanderReward.png")