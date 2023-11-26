import gymnasium as gym
from stable_baselines3 import DQN 
#from stable_baselines.common.vec_env import DummyVecEnv #Dummy Vectorized Environment, some algorithms require a vectorized environment, so this works
from stable_baselines3.common.evaluation import evaluate_policy #Function used to evauluate performance of agent. Calculates Mean and Standard Deviation of all rewards obtained

#BUILDING MODEL
environment_name = 'LunarLander-v2'
env = gym.make(environment_name, render_mode="rgb_array") #Creates Environment
#env = DummyVecEnv([lambda: env]) #Wraps the Environment in the Dummy Vectorized Environment
obs = env.reset()
assert isinstance(env.action_space, gym.spaces.Discrete)
model = DQN('MlpPolicy', env,verbose = 1) # Creates an ACER model with a Multi-Layer Perceptron (MLP) policy 
#When Verbose=1, it prints training info of each episode
model.learn(total_timesteps=10000) # Trains the model for a specified number of timesteps




#TESTING MODEL
evaluate_policy(model, env, n_eval_episodes=10, render=True) #Evaluates 10 episodes and renders the trained model


#SAVING MODEL 
env.close()
model.save("DQN_model")
del model
model = DQN.load("DQN_model", env=env) #Reloading Model to restart
    #We Want High Variance and High Possible Mean Reward

