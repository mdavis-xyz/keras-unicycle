"""
Unicycle model, basically a modified inverted pendulum example
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
        A sp is attached by an un-actuated joint to a wheel, which moves along a frictioned track. The sp starts upright, and the goal is to prevent it from falling over by increasing and reducing the wheel's velocity.

    Observation:
        Type: Box(6)
        0   Wheel Angular Position
                Measured in radians
                0 is
                    wheel at center of screen
                    right pedal at bottom wheel
                    left pedal at top of wheel
                positive means towards the right of the screen, (forward for the rider)
        1   Wheel Angular Velocity (radians/sec)
                Derivative of state 0
        2   sin(state[0])
                this is added because the torques are proportional to sin of the angle
        3   cos(state[0])
                this is added because the torques are proportional to sin of the angle
        4   sp Angular Position
                measured in radians
                0 is vertical
                positive means leaning to the right of the screen
                which is 'forward' from the unicycle rider's perspective
        5   sp angular velocity
                measured in radians/sec

    Actions:
        Type: Discrete(2) # TODO: make continuous
        Num     Action
        0       Push on left pedal
        1       Don't push
        2       Push on right pedal

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the sp is pointing, and the angle of the cranks.

    Reward:
        * +1 for every step taken, including the termination step
        * + [0,0.5] for displacement from center
        * -[0,0.25] if velocity is too high (higher velocity, larger penalty)

    Starting State:
        All observations are assigned a uniform random value close to zero

    Episode Termination:
        Seat Post Angle too large
        Wheel reaches the edge of the display
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 25
    }

    def __init__(self):
        self.gravity = 9.8 # m/s/s
        self.masscart = 1 # kg of wheel set
        self.masspole = 0 # kg of seat post
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # meters, actually half the sp's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = self.gravity * 0.2 * self.total_mass # Newtons, twice the weight of the system
        self.tau = 1.0/self.metadata['video.frames_per_second']  # seconds between state updates
        self.kinematics_integrator = 'euler'


        self.wheel_diameter = 0.4 # meters
        self.wheel_circumference = self.wheel_diameter * math.pi # meters
        self.wheel_radius = self.wheel_diameter / 2.0 # meters

        # Thresholds at which to fail the episode
        self.sp_angle_thresh = math.pi  * 0.4 # radians, seat post
        self.sp_speed_thresh = self.gravity * 3 * math.pi * 2 / self.wheel_circumference # the wheel speed corresponding to a 3 second freefall
        self.wheel_angle_thresh = 3.5 * math.pi # radians, wheel
        self.wheel_speed_thresh = self.gravity * 4 * math.pi * 2 / self.wheel_circumference # the wheel speed corresponding to a 3 second freefall

        # width of the world
        self.world_width = abs((self.wheel_angle_thresh * self.wheel_circumference / (2*math.pi)) + 2*self.wheel_radius)

        # Angle limit set to 2 * sp_angle_thresh so failing observation is still within bounds
        self.limits = np.array([
                self.wheel_angle_thresh, # wheel angle
                self.wheel_speed_thresh, # wheel speed
                1, # sin(wheel_speed)
                1, # cos(wheel_speed)
                self.sp_angle_thresh, # seat post angle
                self.sp_speed_thresh # seat post velocity
            ], dtype=np.float32)

        # action | meaning
        # 0      | push down left pedal
        # 1      | nothing
        # 2      | push down right pedal
        self.action_space_size = 3 # number of choices
        self.action_offset = self.action_space_size / 2.0
        self.action_space = spaces.Discrete(self.action_space_size)

        self.high = self.limits / self.limits # normalized limits
        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # takes in an action
    # returns the horizontal force in Netwons exerted from ground to wheel, towards the left?
    def action_to_force(self,action):
        # if wheel angle == 0, pedals are at bottom/top, so no torque, so no horizontal force
        # if wheel angle == 90 degrees, maximum torque, so maximum horizontal force
        wheel_angle = self.state[0]*self.limits[0] # un-normalize to radians
        if action == 0: # push down left pedal
            force = -math.sin(wheel_angle) * self.force_mag
        if action == 1: # no pushing
            force = 0
        else: # push down right pedal
            force = math.sin(wheel_angle) * self.force_mag
        return(force)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.last_action = action # for rendering
        state = self.state * self.limits
        x = self.wheel_circumference * state[0] / (2*math.pi) # convert rad to m
        x_dot = self.wheel_circumference * state[1] / (2*math.pi) # convert rad/s to m/s
        theta = state[4]
        theta_dot = state[5]
        if action == 0:
            force = -1 * self.force_mag
        elif action == 1:
            force = 0
        else:
            force = 1 * self.force_mag
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
        wheel_angle = 2*math.pi * x / self.wheel_circumference
        wheel_angle_dot = 2*math.pi * x_dot / self.wheel_circumference
        self.state = (wheel_angle,
                      wheel_angle_dot,
                      math.sin(wheel_angle),
                      math.cos(wheel_angle),
                      theta,
                      theta_dot
                      ) / self.limits

        done =  False #(np.absolute(new_state_unnorm) > self.limits).any()

        # if done:
        #     if abs(new_state_unnorm[4]) < 1:
        #         print("Failing dimension not sp angle: " + str(new_state_norm >= 1))

        if not done:
            reward = 1.0 # staying up is most important

            # slight reward for staying in center
            # 0.5 if in center
            # 0 at edge
            # linear (TODO: make non-linear)
            x_norm = self.state[0] # -1 to 1
            reward += 0.2 * (1 - abs(x_norm))

            # slight penalty for having wheel velocity high
            # relu penalty
            x_dot_norm = self.state[1] # -1 to 1
            x_dot_penalty_thresh = 0.4
            if x_dot_norm > x_dot_penalty_thresh:
                print("horizontal speed too high, penalizing")
                reward -= 0.2 * (abs(x_dot_norm) - x_dot_penalty_thresh) / (1-x_dot_penalty_thresh)

            # slight penalty for having seatpost angle to large
            sp_angle_norm = self.state[4] # -1 to 1
            reward -= 0.2 * (1 - abs(sp_angle_norm))

            # slight penalty for having seatpost move too fast, but only if moving away from center (getting worse)
            sp_angle_vel_norm = self.state[5]
            if sp_angle_norm*sp_angle_vel_norm > 0: # both variables have same sign
                # seat is falling forwards while moving forwards
                # or seat is falling back while moving back
                reward -= 0.1 *  (1 - abs(sp_angle_vel_norm))


        elif self.steps_beyond_done is None:
            # sp just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        wheel_angle = random.uniform(-math.pi/2.0,math.pi/2.0)
        wheel_angle_vel = random.uniform(-math.pi * 2 / 2,math.pi * 2 / 2) * 0 # 1 rotation each 2 seconds
        sp_angle = random.uniform(-math.pi/8.0, math.pi/8.0)
        sp_angle_vel_temp = random.uniform(-math.pi * 2 / 5,math.pi * 2 / 5) * 0 # 1 rotation each 5 seconds
        self.state = np.array([
            wheel_angle,
            wheel_angle_vel,
            math.sin(wheel_angle),
            math.cos(wheel_angle),
            sp_angle,
            sp_angle_vel_temp
        ], dtype=np.float32) / self.limits

        self.steps_beyond_done = None
        self.last_action = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 1000
        screen_height = 500


        scale = screen_width/self.world_width
        carty = 100 # TOP OF CART
        spwidth = 10.0
        splen = scale * (2 * self.length)
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
                'left': {
                    'trans':rendering.Transform(),
                    'action':0,
                    'multiplier':1
                },
                'right': {
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

            # sp
            l,r,t,b = -spwidth/2,spwidth/2,splen-spwidth/2,-spwidth/2
            sp = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            sp.set_color(.8,.6,.4)
            self.sptrans = rendering.Transform()
            sp.add_attr(self.sptrans)
            sp.add_attr(self.carttrans)
            self.viewer.add_geom(sp)
            self.axle = rendering.make_circle(spwidth/2)
            self.axle.add_attr(self.sptrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,floor_y), (screen_width,floor_y))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._sp_geom = sp

        if self.state is None:
            return None

        # Edit the sp polygon vertex
        sp = self._sp_geom
        l,r,t,b = -spwidth/2,spwidth/2,splen-spwidth/2,-spwidth/2
        sp.v = [(l,b), (l,t), (r,t), (r,b)]

        state_unnorm = self.state * self.limits

        x = (state_unnorm[0] / (2*math.pi)) * self.wheel_circumference # meters
        x_dr = x*scale + screen_width/2.0 # 0 is middle of screen
        self.carttrans.set_translation(x_dr, wheel_radius_dr+floor_y)

        # pedals
        for p in self.pedals:
           # 0 radians means pedals vertical
           # and wheelAngle()==0 means left pedal at top
           # increasing state[0] means wheel turns clockwise
           m = self.pedals[p]['multiplier']
           pedal_x = m*math.sin(state_unnorm[0])*crank_len
           pedal_y = m*math.cos(state_unnorm[0])*crank_len
           self.pedals[p]['trans'].set_translation(pedal_x+x_dr,pedal_y+floor_y+wheel_radius_dr)
           if self.last_action == self.pedals[p]['action']:
               pedal_active_col(self.pedals[p]['pedal'])
           else:
               pedal_inactive_col(self.pedals[p]['pedal'])
        self.sptrans.set_rotation(state_unnorm[4])
        if mode == 'human':
            time.sleep(1.0 / self.metadata['video.frames_per_second'])
        return self.viewer.render(return_rgb_array = False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
