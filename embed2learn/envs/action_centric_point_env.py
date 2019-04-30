from enum import Enum

import gym
from garage.core import Serializable
import numpy as np

from embed2learn.envs.point_env import PointEnv

class Direction(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

# Smaller set of starting point
# makes training faster since there's
# bug in this implementation
STARTINGPOINTS = [
    (0., 0.),
    (2., 0.),
    (0., -2.),
    (-2., 0.),
    (0., 2.),
]


class ActionCentricPointEnv(PointEnv):

    def __init__(
            self,
            direction=Direction.UP,  # TODO: Change this to angle instead of discrete direction
            distance=3.,
            *args,
            **kwargs):
        Serializable.quick_init(self, locals())

        self._init_point = np.array([0., 0.])
        self._point = np.copy(self._init_point)
        self._direction = direction
        self._distance = distance

        # build a auxilary goal here
        # the goal will be changed during reset
        goal = self._get_goal()
        kwargs['goal'] = goal
        super().__init__(*args, **kwargs)

    def _get_goal(self):
        if self._direction == Direction.UP:
            direction = np.array([0., 1.])
        elif self._direction == Direction.RIGHT:
            direction = np.array([1., 0])
        elif self._direction == Direction.LEFT:
            direction = np.array([-1, 0.])
        else:
            direction = np.array([0., -1.])
        goal = self._init_point + direction * self._distance
        return goal

    def reset(self):
        # Reset the initial position first
        # Make trajs concatenatable
        global STARTINGPOINTS
        random_idx = np.random.randint(low=0, high=len(STARTINGPOINTS))
        self._init_point = STARTINGPOINTS[random_idx] + np.random.normal(scale=0.05, size=2)
        self._point = np.copy(self._init_point)
        # Set the goal based on the init positions
        self._goal = self._get_goal()
        self._traces.append([tuple(self._point)])
        return np.concatenate([self._point, self._init_point], axis=0)

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4, ), dtype=np.float32)

    def step(self, action):
        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a *= self._action_scale
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self._point = np.clip(self._point + a, -self.max_reach_range, self.max_reach_range)
        self._traces[-1].append(tuple(self._point))

        dist = np.linalg.norm(self._point - self._goal)
        done = dist < np.linalg.norm(self.action_space.low)

        # dense reward
        reward = -dist
        is_success = False
        # completion bonus
        if done:
            is_success = True
            reward += self._completion_bonus

        # sometimes we don't want to terminate
        done = done and not self._never_done
        return np.concatenate([self._point, self._init_point], axis=0), reward, done, dict(is_success=is_success)
