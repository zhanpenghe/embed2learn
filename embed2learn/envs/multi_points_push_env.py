import os
import numpy as np

from garage.core import Serializable

import gym
from gym import utils
from gym.envs.mujoco import mujoco_env

from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm

import itertools

# objects defined in xml
N = 4
objects = [
    "red",    # rgba 1  0  0 1
    "green",  # rgba 0  .8 0 1
    "yellow", # rgba 1  1  0 1
    "blue",   # rgba .3 .3 1 1
]

TASKS = [
    {'goal_description': [1, 3], 'goal_label': 'red'},
    {'goal_description': [1, 4], 'goal_label': 'green'},
    {'goal_description': [1, 2], 'goal_label': 'yellow'},
    {'goal_description': [1, 5], 'goal_label': 'blue'},
]

def code_to_one_hot_matrix(code, sentence_code_dim):
    n_vocab = sentence_code_dim
    oh_encoding = np.zeros(
        shape=(len(code), n_vocab),
        dtype=np.float32
    )
    oh_encoding[np.arange(len(code)), code] = 1.
    return oh_encoding

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def dist_point2seg(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)

class MultiPointsPushEnv(mujoco_env.MujocoEnv, utils.EzPickle, Serializable):
    def __init__(
                self,
                tasks=TASKS,
                objects=objects,
                # dictionary=None,
                # task_env_cls=None,
                # task_args=None,
                # task_kwargs=None,
        ):

        # initialize ezpickle
        utils.EzPickle.__init__(self)

        # prev action
        self.prev_action = None


        # TODO: hard code zone
        # - num_tasks
        self.num_tasks = 4 # TODO
        self.num_objects = N # TODO
        self.tasks = tasks
        self.max_sentence_length = 2
        self.code_sentence_dim = 6

        
        # Task Selection
        self.active_task = None
        self._task_selection = itertools.cycle(range(self.num_tasks))

        # Helper
        # self.vectorizer = vectorizer

        # Process Objects
        self.objects = objects
        self.object_name_id = {}
        self.id_object_name = {}
        for i, name in enumerate(objects, start=1):
            self.object_name_id[name] = i
            self.id_object_name[i] = name

        self._all_one_hots = None

        # initialize environment
        mujoco_path = os.path.join(os.path.dirname(__file__), 'assets', 'multi_points_pusher.xml')
        frame_skip = 5
        mujoco_env.MujocoEnv.__init__(self, mujoco_path, frame_skip)


    # New!
    @property
    def all_task_one_hots(self):
        if self._all_one_hots is None:
            self._all_one_hots = [
                code_to_one_hot_matrix(t['goal_description'], self.code_sentence_dim)
                for t in self.tasks
            ]
        return self._all_one_hots


    # New!
    def reset(self):
        self.active_task = next(self._task_selection)
        return super().reset()

    # New!
    @property
    def active_task_one_hot_gt(self):
        one_hot = np.zeros(self.num_objects)
        t = 0 if self.active_task is None else self.active_task
        one_hot[t] = 1
        return one_hot

    # New!
    @property
    def active_task_one_hot(self):
        t = 0 if self.active_task is None else self.active_task
        goal_description = self.tasks[t]["goal_description"]
        # goal_description = self._task_envs[t]._goal_description
        one_hot = code_to_one_hot_matrix(goal_description, self.code_sentence_dim)
        return one_hot

    # New!
    @property
    def task_space(self):
        n = self.max_sentence_length
        one_hot_ub = np.ones(n) * (self.code_sentence_dim - 1)
        one_hot_lb = np.zeros(n)
        return gym.spaces.Box(one_hot_lb, one_hot_ub, dtype=np.float32)

    def _loc(self, obj_name):
        return self.get_body_com(obj_name)

    def _loc2(self, obj_name):
        return np.array(self._loc(obj_name))[:2]

    # TODO
    """
    reward for active task to its object
    """
    def step(self, action, debug=False):
        # TODO
        t = 0 if self.active_task is None else self.active_task
        task = self.tasks[t]


        # print object states before stepping
        if debug:
            print("Printing object states.")
            print("Robot Arm: {}".format(self._loc("robot")))
            print("Left Gripper: {}".format(self._loc("l_hand")))
            print("Right Gripper: {}".format(self._loc("r_hand")))
            print("Object 1: {}".format(self._loc("obj1")))
            print("Goal 1: {}".format(self._loc("goal1")))

        # calculate reward
        reward = 0
        arm_location = self._loc2("robot")
        for obj in range(1, N + 1):
            locs = {}
            for name in ['goal', 'obj']:
                locs[name + '_joint'] = self._loc2("{}{}".format(name, obj))
                locs[name + '_center'] = (locs[name + '_joint'])

            obj2target = lambda goal, obj: obj + (obj - goal) / norm(obj - goal) * 0.08

            targetj = obj2target(locs['goal_joint'], locs['obj_joint'])

            dist_arm2seg = dist_point2seg(targetj, targetj, arm_location)

            # Change
            # reward only current active task
            if obj == self.object_name_id[task['goal_label']]:
                print(obj, task) 
                # part 1: distance between object part and goal
                reward -= np.linalg.norm(locs['goal_center'] - locs['obj_center'])

                # part 2: distance between arm and object part
                reward -= max(0, dist_arm2seg - 0.04)

        # part 3: arm preferred to be on the ground
        reward -= max(0, self.get_body_com('robot')[2] - 1.12) * 0.2

        # part 4: penalize action
        # reward -= norm(action) * 0.1 

        # part 5: penalize non-smooth action
        # if self.prev_action is not None:
        #     reward -= norm(action - self.prev_action) * 0.1
        # self.prev_action = action

        # step the environment
        self.do_simulation(action, self.frame_skip)


        # get observation
        obs = self._get_obs()


        # get if episode completed or not
        done = False

        # generate info
        info = 'keep going' if reward < -0.1 else 'good job'

        return obs, reward, done, dict(info=info)

    def viewer_setup(self):
        # camera distance
        self.viewer.cam.distance = 2
        
        # viewing angle
        self.viewer.cam.lookat[1] = -0.5
        self.viewer.cam.lookat[2] = 1.0

    def reset_model(self):
        qpos = self.init_qpos

        self.prev_action = None

        goal_obj_pos_arr = []
        for i in range(N):
            goal_obj_pos_arr.append(0)
            while True:
                # object and goal should have the same rotation
                rotation = np.random.uniform(low=-31.4, high=31.4, size=1)
                obj_pos = np.concatenate([
                    rotation,
                    np.random.uniform(low=-0.6, high=0.6, size=1),
                    np.random.uniform(low=-0.6, high=0.6, size=1)
                ])
                goal_pos = np.concatenate([
                    rotation,
                    np.random.uniform(low=-0.6, high=0.6, size=1),
                    np.random.uniform(low=-0.6, high=0.6, size=1)
                ])
                # constrain distance to (0.1, 0.3)
                if np.abs(np.linalg.norm(goal_pos[1:] - obj_pos[1:]) - 0.2) < 0.1:
                    goal_obj_pos_arr.extend(obj_pos.tolist())
                    goal_obj_pos_arr.extend(goal_pos.tolist())
                    break
        
        qpos[-len(goal_obj_pos_arr):] = goal_obj_pos_arr

        qvel = self.init_qvel + np.random.uniform(low=-0.005, high=0.005,
                        size=self.model.nv)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:5],
            self.sim.data.qvel.flat[:5],
            self._loc('robot'),
            self._loc('obj1'),
            self._loc('goal1'),
            self._loc('obj2'),
            self._loc('goal2'),
            self._loc('obj3'),
            self._loc('goal3'),
            self._loc('obj4'),
            self._loc('goal4'),
        ])
