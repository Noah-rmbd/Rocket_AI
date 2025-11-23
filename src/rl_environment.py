import numpy as np
from environment import Environment

class RLEnvironment:
    def __init__(self):
        self.env = Environment()
        # Action space: 8 combinations of 3 engines
        # 0: 000, 1: 001, 2: 010, 3: 011, 4: 100, 5: 101, 6: 110, 7: 111
        self.action_space_n = 8
        self.state_space_n = 3 # cos(theta), sin(theta), angular_velocity
        
        # Mapping action index to engine states (e1, e2, e3)
        self.action_map = {
            0: (False, False, False),
            1: (False, False, True),
            2: (False, True, False),
            3: (False, True, True),
            4: (True, False, False),
            5: (True, False, True),
            6: (True, True, False),
            7: (True, True, True)
        }
        
        self.max_steps = 600 # 10 seconds at 60fps
        self.current_step = 0

    def reset(self):
        self.env.rocket.reset_position()
        # Randomize initial orientation slightly to force learning stabilization
        self.env.rocket.orientation = np.random.uniform(-0.5, 0.5) 
        self.env.rocket.angular_velocity = np.random.uniform(-1.0, 1.0)
        self.env.rocket.velocity = np.zeros(2)
        
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        # State: [cos(theta), sin(theta), angular_velocity]
        theta = self.env.rocket.orientation
        av = self.env.rocket.angular_velocity
        return np.array([np.cos(theta), np.sin(theta), av], dtype=np.float32)

    def step(self, action_idx):
        e1, e2, e3 = self.action_map[action_idx]
        
        # Fixed time step
        dt = 1/60.0
        self.env.iterate(dt, e1, e2, e3)
        
        state = self._get_state()
        
        # Reward function
        # Goal: Minimize angular velocity and keep angle close to 0 (upright)
        # Upright means theta = 0 -> cos(theta) = 1
        
        theta = self.env.rocket.orientation
        av = self.env.rocket.angular_velocity
        
        # Normalize angle to [-pi, pi] for reward calculation if needed, 
        # but cos/sin handle it naturally.
        # We want cos(theta) -> 1.
        
        r_angle = (np.cos(theta) - 1.0) # 0 is best, -2 is worst (upside down)
        r_av = -0.1 * abs(av) # Penalty for spinning
        
        reward = r_angle + r_av
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return state, reward, done, {}
