# Machine Learning For Unicycling

This project is for training a neural network to ride a sumlation of a unicycle, using reinforcement learning.

![Video of simulated unicycle riding](trained/videos/best.mp4)

I took the cartpole example from [here](https://github.com/keras-rl/keras-rl), and modified to be a unicycle.

I find the gym+keras ecosystem terribly confusing. Having a custom environment is really confusing, in terms of directory structure. So the `env.py` is supposed to be a shortcut to `gym-unicycle/gym_unicycle/envs/unicycle_env.py`. But sometimes shortcuts don't work well with git.

## How to Use

* Run `./makeEnv.sh`. This creates a virtual env. I can't have just a `requirements.txt` file, because the installation of gym and a custom env is a bit odd.
