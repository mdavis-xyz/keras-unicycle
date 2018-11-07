from gym.envs.registration import register

register(
    id='Unicycle-v0',
    entry_point='gym_unicycle.envs:UnicyleEnv',
    max_episode_steps=200,
    reward_threshold=195,
)
