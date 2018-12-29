import numpy as np
from multiprocessing import Pool
import gym
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from gym.envs.registration import registry, register, make, spec
import gym_unicycle
#from gym import wrappers # xvfb-run -s "-screen 0 1400x900x24" python <your_script.py>
import os.path
from time import time
from statistics import mean
import pprint as pp

ENV_NAME = 'MATTENV-v0'
register(
    id=ENV_NAME,
    entry_point='gym_unicycle.envs:UnicycleEnv',
    max_episode_steps=2000,
    reward_threshold=800.0,
)


def attempt(args):
    lr = args['lr']
    nb_steps = args['nb_steps']
    activation = args['activation']
    layerType = args['layerType']
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    #env = wrappers.Monitor(env, './videos/' + str(time()) + '/', force=True)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    print("env.observation_space.shape: " + str(env.observation_space.shape))

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    if layerType == 0:
        model.add(Dense(16))
        model.add(Activation(activation))
        model.add(Dense(16))
        model.add(Activation(activation))
        model.add(Dense(16))
        model.add(Activation(activation))
    elif layerType == 1:
        model.add(Dense(16))
        model.add(Activation(activation))
        model.add(Dense(13))
        model.add(Activation(activation))
        model.add(Dense(10))
        model.add(Activation(activation))
    else:
        model.add(Dense(16))
        model.add(Activation(activation))
        model.add(Dense(13))
        model.add(Activation(activation))
        model.add(Dense(3))
        model.add(Activation(activation))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    if not os.path.exists('weights'):
        os.makedirs('weights')
    weights_fname = 'weights/dqn_%s_%f_%d_%d_weights.h5f' % (ENV_NAME,lr,nb_steps,layerType)
    if os.path.isfile(weights_fname):
        print("Loading weights from before")
        print("Skipping training")
        dqn.load_weights(weights_fname)
    else:
        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=1)

        # After training is done, we save the final weights.
        dqn.save_weights(weights_fname, overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    env.reset()
    env.close()
    result = dqn.test(env, nb_episodes=5, visualize=False)
    means = {
        'reward': mean(result.history['episode_reward']),
        'steps': mean(result.history['nb_steps'])
            }
    with open(weights_fname.replace('h5f','json'),"w") as f:
            f.write(result.history)
    return(means) 

def main():
    args = []
    for lr in [5e-3, 2e-3, 1e-3, 5e-4, 1e-4]:
        for nb_steps in [10**x for x in range(4,7)]:
            for activation in ['tanh','relu']:
                for layerType in range(3):
                    arg = {
                        'lr':lr,
                        'nb_steps':nb_steps,
                        'activation':activation,
                        'layerType':layerType
                        }
                    args.append(arg)

    pp.pprint(args)
    with Pool(6) as p:
        results = p.map(attempt, args)
    data = [{**a,**r} for (a,r) in zip(args,results)]
    data.sort(key=lambda x: x['steps'])
    pp.pprint(data)

main()
