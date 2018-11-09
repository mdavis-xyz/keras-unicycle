"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import random
import time

class UnicycleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num     Observation                 Min         Max
        0       Cart Position             -4.8            4.8
        1       Cart Velocity             -Inf            Inf
        2       Pole Angle                 -24°           24°
        3       Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num     Action
        0       Push cart to the left
        1       Push cart to the right
        
        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 25 
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 12.0 # originally 10
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'


        self.wheel_diameter = 0.5 
        self.wheel_circumference = self.wheel_diameter * math.pi
        self.wheel_radius = self.wheel_diameter / 2.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 45 * 2 * math.pi / 360
        self.x_threshold = 2.4 # TODO: correct for angle
        self.wa_threshold = self.xToWheelAngle(self.x_threshold)

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.wa_threshold * 4,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
            float("inf"), # cos(wheel angle)
            float("inf")  # sin(wheel angle)
            ])

        #self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # float from -1 to 1
        # left, nothing, right
        
        # action | meaning
        # 0      | push down pedal A
        # 1      | nothing
        # 2      | push down pedal B
        self.action_space_size = 3 # number of choices
        self.action_offset = self.action_space_size / 2.0
        self.action_space = spaces.Discrete(self.action_space_size)
        assert(self.normalize_action(0) == -1)
        assert(self.normalize_action(self.action_space_size) == 1)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def xToWheelAngle(self,x):
        return(x / self.wheel_radius)

    # in radians
    # 0 radians means pedals vertical
    def wheelAngleToX(self,angle):
        return(angle * self.wheel_radius)


    # I want to have actions from -1 to 1
    # but for some reason it does 0 to self.action_space_size 
    # so I need to map 0..action_space_size to -1..1
    def normalize_action(self,action):
        shifted = action - self.action_offset
        scaled = shifted / (self.action_space_size / 2.0)
        return(scaled)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x = self.wheelAngleToX(state[0])
        x_dot = self.wheelAngleToX(state[1])
        theta = state[2] # seat post angle
        theta_dot = state[3] 

        # if angle == 0, pedal is at bottom/top, so no torque, so no horizontal force
        # if angle == 90 degrees, maximum torque, so maximum horizontal force
        if action == 0: # push down pedal A
            force = math.sin(self.state[0]) * self.normalize_action(action) * self.force_mag
        if action == 1: # no pushing
            force = 0
        else: # push down pedal B
            force = -math.sin(self.state[0]) * self.normalize_action(action) * self.force_mag
        self.last_action = action # for rendering
        #force = self.normalize_action(action) * self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        wa = self.xToWheelAngle(x)
        wa_dot = self.xToWheelAngle(x_dot)
        self.state = (wa,wa_dot,theta,theta_dot,math.cos(wa),math.sin(wa))
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0

            # slight reward for staying in center
            # 0.5 if in center
            # 0 at edge
            # linear (TODO: make non-linear)
            reward += (self.x_threshold - abs(x) )/ float(self.x_threshold*2.0)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.state[0] += random.uniform(-math.pi/2.0,math.pi/2.0)
        self.state[4] = math.cos(self.state[0])
        self.state[5] = math.sin(self.state[0])
        #self.state[1] = 0 # no horizontal velocity
        self.steps_beyond_done = None
        self.last_action = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 1200
        screen_height = 400

        world_width = self.x_threshold*2+self.wheel_diameter
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        #cartwidth = 50.0
        #cartheight = 30.0
        wheel_radius_dr = self.wheel_radius * scale
        wheel_diameter_dr = self.wheel_diameter * scale
        wheel_circumference_dr = self.wheel_circumference * scale
        crank_len = wheel_radius_dr * 0.7
        pedal_width = scale * 0.15
        pedal_thick = scale * 0.03
        floor_y = 30.0

        pedal_inactive_col = lambda pedal: pedal.set_color(.3,.4,.5)
        pedal_active_col = lambda pedal: pedal.set_color(.8,.0,.5)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            #cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            cart = self.viewer.draw_circle(wheel_diameter_dr/2.0,filled=False)
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            
            # pedals
            self.pedal_trans = {}
            self.pedals = {
                'A': {
                    'trans':rendering.Transform(),
                    'action':0,
                    'multiplier':1
                },
                'B': {
                    'trans':rendering.Transform(),
                    'action':2,
                    'multiplier':-1
                }
            }
            for p in self.pedals:
                l,r,t,b = -pedal_width/2, pedal_width/2, pedal_thick/2, -pedal_thick/2
                self.pedals[p]['pedal'] = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                self.pedals[p]['pedal'].add_attr(self.pedals[p]['trans'])
                pedal_inactive_col(self.pedals[p]['pedal'])
                self.viewer.add_geom(self.pedals[p]['pedal'])

            # seatpost
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform()
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,floor_y), (screen_width,floor_y))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, wheel_radius_dr+floor_y)

        # pedals
        for p in self.pedals:
           # 0 radians means pedals vertical
           # and wheelAngle()==0 means wheel A at top
           # increasing state[0] means wheel turns clockwise
           m = self.pedals[p]['multiplier']
           pedal_x = m*math.sin(self.state[0])*crank_len
           pedal_y = m*math.cos(self.state[0])*crank_len
           self.pedals[p]['trans'].set_translation(pedal_x+cartx,pedal_y+floor_y+wheel_radius_dr)
           if self.last_action == self.pedals[p]['action']:
               pedal_active_col(self.pedals[p]['pedal'])
           else:
               pedal_inactive_col(self.pedals[p]['pedal'])
        self.poletrans.set_rotation(-x[2])
        if mode == 'human':
            time.sleep(1.0 / self.metadata['video.frames_per_second'])
        return self.viewer.render(return_rgb_array = False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
