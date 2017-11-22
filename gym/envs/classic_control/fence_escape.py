import gym
from gym import spaces
import numpy as np
from os import path
import math

class FenceEscapeEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.dt = 0.01

        self.action_space = spaces.Box(-1, 1, (1,))
        self.observation_space = spaces.Box(-1, 11, (2,))

        self.viewer = None
        
        self.state = self.observation_space.sample()
        
    def _step(self, u):
        self.state = self.observation_space.sample()
        reward = -1.0
        done = False

        return self.state, reward, done, {}

    def _reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def _get_obs(self):
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            
            self.viewer = rendering.Viewer(750, 500)
            self.viewer.set_bounds(-2, 12, 0, 5)
            
            evader = rendering.make_circle(1, 30, True)
            evader.set_color(0,1,0)
            self.evader_transform = rendering.Transform()
            evader.add_attr(self.evader_transform)
            self.viewer.add_geom(evader)
            
            pursuer = rendering.make_circle(1, 30, True)
            pursuer.set_color(1,0,0)
            self.pursuer_transform = rendering.Transform()
            pursuer.add_attr(self.pursuer_transform)
            self.viewer.add_geom(pursuer)

            fence = rendering.Line((0,2.5), (10,2.5))
            self.viewer.add_geom(fence)
            
        # draw
        self.viewer.render()
        self.evader_transform.set_translation(self.state[0], 3.75)
        self.pursuer_transform.set_translation(self.state[1], 1.25)
        if mode == 'rgb_array':
            return self.viewer.get_array()
        elif mode == 'human':
            pass
        else:
            return super(FenceEscape, self).render(mode=mode)
                

    
    
