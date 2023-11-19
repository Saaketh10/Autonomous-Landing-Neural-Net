import gym
from stable_baselines import ACER
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy

environment_name = 'LunarLander-v2'
env = gym.make(environment_name)

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        score += reward

    print('Episode:{} Score:{}'.format(episode, score))

env.close()

