from gym.envs.registration import register
register(
    id='MoonLanding-v0',
    entry_point='gym_foo.envs:MoonLandingEnv',
    max_episode_steps=100,
)
