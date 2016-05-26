import gym
from gym import spaces
import numpy as np
from os import path
import math
import random

'''
Based on the Quadrotor2D implementation in Drake:
https://github.com/RobotLocomotion/drake/tree/master/drake/examples/Quadrotor2D

The configuration space consists of:
q(0) - x position
q(1) - z position
q(2) - theta

Since we're dealing with a second-order system, that means that the
state also consists of the derivatives of q(0) through q(2).

The actions are:
u(0) - thrust for left prop
u(1) - thrust for right prop
'''

class PlanarQuadEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 60
        }

    def __init__(self):
        self.dt = 0.02
        self.viewer = None

        self.done = False
        self.success_count = 0

        self.action_space = spaces.Box(-5, 5, (2,))
        self.observation_space = spaces.Box(low=np.asarray([-10, 0, 0, 0, 0, 0]), high=np.asarray([10, 10, 0, 0, 0 ,0]))

    def _step(self, u):
        # based on Drake, which is based on (Bouadi, Bouchoucha, Tadjine 2007)
        L = 0.25 # length of rotor arm
        m = 0.486 # mass of the quadrotor
        I = 0.00383 # moment of inertia
        g = 9.81 # gravity

        u = np.clip(u, 0, 10)
        
        pos = self.state[0:3]
        vel = self.state[3:6]

        # Stormer-Verlet integration
        qdd = lambda pos: np.asarray([-math.sin(pos[2])/m * (u[0] + u[1]),
                                      -g + math.cos(pos[2])/m * (u[0] + u[1]),
                                      L/I * (-u[0] + u[1])])
        v_t_plus_one_half = vel + 0.5 * qdd(pos) * self.dt
        x_t_plus_dt = pos + v_t_plus_one_half * self.dt
        a_t_plus_dt = qdd(x_t_plus_dt)
        v_t_plus_dt = v_t_plus_one_half + 0.5 * a_t_plus_dt * self.dt

        self.state = np.concatenate((x_t_plus_dt, v_t_plus_dt), axis=0)

        if ( (pos[0] - 0)**2 + (pos[1] - 12)**2 < 0.1 ):
            reward = 10.0
            self.success_count += 1
            if self.success_count > 60:
                self.done = True
        elif pos[1] < 0 or pos[1] > 22 or abs(pos[0]) > 12:
            self.done = True
            reward = -1.0
        else:
            self.done = False
            reward = -1.0

        return self.state, reward, self.done, {}
        

    def _reset(self):
        self.done = False
        self.state = np.asarray([random.random() * 16 - 8,
                                 random.random() * 16,
                                 0, 0,0,0]) + np.concatenate((np.zeros(3),
                                                              np.random.randn(3)),
                                                             axis=0)
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(600, 600)
            self.viewer.set_bounds(-10, 10, 0, 20)

            airframe = rendering.make_capsule(2.5,0.15)
	    airframe.add_attr(rendering.Transform(translation=(-1.25, 0.0)))
            airframe.set_color(0,0,1)
            self.airframe_transform = rendering.Transform()
            airframe.add_attr(self.airframe_transform)

            left_prop_shaft = rendering.Line((0,0), (0,0.25))
            left_prop_shaft.add_attr(rendering.Transform(translation=(-0.875, 0.0)))
            left_prop_shaft.add_attr(self.airframe_transform)
            self.viewer.add_geom(left_prop_shaft)

            left_prop = rendering.make_capsule(0.75, 0.1)
            left_prop.add_attr(rendering.Transform(translation=(-1.25, 0.25)))
            left_prop.add_attr(self.airframe_transform)
            left_prop.set_color(0,1,0)
            self.viewer.add_geom(left_prop)

            right_prop_shaft = rendering.Line((0,0), (0,0.25))
            right_prop_shaft.add_attr(rendering.Transform(translation=(0.875, 0.0)))
            right_prop_shaft.add_attr(self.airframe_transform)
            self.viewer.add_geom(right_prop_shaft)

            right_prop = rendering.make_capsule(0.75, 0.1)
            right_prop.add_attr(rendering.Transform(translation=(0.5, 0.25)))
            right_prop.add_attr(self.airframe_transform)
            right_prop.set_color(0,1,0)
            self.viewer.add_geom(right_prop)

            self.viewer.add_geom(airframe)

        q = self.state
        self.airframe_transform.set_translation(q[0], q[1])
        self.airframe_transform.set_rotation(q[2])

        self.viewer.render()
        if mode == 'rgb_array':
            return self.viewer.get_array()
        elif mode is 'human':
            pass
        else:
            return super(PlanarQuadEnv, self).render(mode=mode)
