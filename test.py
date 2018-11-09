import argparse
import sys

import gym
from gym import wrappers, logger

from gym.envs.registration import registry, register, make, spec
import gym_unicycle
import os.path


ENV_NAME = 'MATTENV-v0'
register(
    id=ENV_NAME,
    entry_point='gym_unicycle.envs:UnicycleEnv',
    max_episode_steps=2000,
    reward_threshold=195.0,
)




env = gym.make(ENV_NAME)
env.reset()
for i in range(500):
    env.render(mode='human')
    action = int(i / 80) % 3
    if not env.action_space.contains(action):
        print("Action: " + str(action))
    assert(env.action_space.contains(action))
    env.step(action)
env.env.close()
