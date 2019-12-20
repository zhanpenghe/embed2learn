import sys
import time

import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from embed2learn.envs.util import colormap_mpl
from embed2learn.envs.multi_task_env import normalize, TfEnv


MAX_PATH_LENGTH = 50
SAMPLING_POSITIONS = np.linspace(-1, 1, num=30)
WINDOW_SIZE = 4


def rollout_given_z(env,
                    agent,
                    z,
                    max_path_length=np.inf,
                    animated=False,
                    speedup=1):
    o = env.reset()
    agent.reset()

    if animated:
        env.render()

    path_length = 0
    observations = []
    while path_length < max_path_length:
        a, agent_info = agent.get_action_from_latent(z, o)
        next_o, r, d, env_info = env.step(a)
        observations.append(agent.observation_space.flatten(o))
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            time.sleep(0.05 / speedup)

    return np.array(observations)


def rollout_imitation(env,
                    agent,
                    traj,
                    inference,
                    max_path_length=np.inf,
                    animated=False,
                    speedup=1):
    o = env.reset()
    agent.reset()

    if animated:
        env.render()

    path_length = 0
    observations = []
    while path_length < max_path_length:
        if path_length + WINDOW_SIZE >= traj.shape[0]:
            window = traj[traj.shape[0] - WINDOW_SIZE: traj.shape[0], ...]
        else:
            window = traj[path_length: path_length + WINDOW_SIZE, ...]
        z, z_info = inference.get_latent(window)
        a, agent_info = agent.get_action_from_latent(z, o)
        next_o, r, d, env_info = env.step(a)
        observations.append(agent.observation_space.flatten(o))
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            time.sleep(0.05 / speedup)

    return np.array(observations)


def get_z_dist(t, policy):
    """ Get the latent distribution for a task """
    onehot = np.zeros(policy.task_space.shape, dtype=np.float32)
    onehot[t] = 1
    _, latent_info = policy.get_latent(onehot)
    return latent_info


def imitate(pkl_filename):
    with tf.Session():
        # Unpack the snapshot
        snapshot = joblib.load(pkl_filename)
        env = snapshot['env']
        policy = snapshot['policy']
        inference_net = snapshot['inference']

        # collect expert data
        task_envs = env.env._task_envs
        num_tasks = len(task_envs)
        goals = np.array([te._goal for te in task_envs])

        task_cmap = colormap_mpl(num_tasks)
        expert_data = []
        imitation_data = []

        for t, env in enumerate(task_envs):
            # Get latent distribution
            infos = get_z_dist(t, policy)
            z_mean, z_std = infos['mean'], np.exp(infos['log_std'])

            plt.scatter(
                [goals[t, 0]* 3] , [goals[t, 1]*3],
                s=50,
                color=task_cmap[t],
                zorder=2,
                label="Task {}".format((t + 1)))

            for i, x in enumerate(SAMPLING_POSITIONS):
                # systematic sampling of latent from embedding distribution
                z = z_mean + x * z_std

                # Run rollout
                obs = rollout_given_z(
                    TfEnv(env),
                    policy,
                    z,
                    max_path_length=MAX_PATH_LENGTH,
                    animated=False)
                expert_data.append(obs)

                obs_imitate = rollout_imitation(
                    TfEnv(env),
                    policy,
                    obs,
                    inference_net,
                    max_path_length=MAX_PATH_LENGTH,
                    animated=False,)
                imitation_data.append(obs_imitate)

                # Plot rollout
                plt.plot(obs[:, 0], obs[:, 1], alpha=0.3, color=task_cmap[t])
                plt.plot(obs_imitate[:, 0], obs_imitate[:, 1], alpha=1, color=task_cmap[t])

        plt.grid(True)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.axes().set_aspect('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig('rollout.pdf')
        plt.show()

        expert_data = np.array(expert_data)
        imitation_data = np.array(imitation_data)
        distance = np.mean(
            np.sum(np.square(expert_data - imitation_data), axis=1)
        )
        print("error: {}".format(distance))
        

def imitate_unseen(pkl_filename):
    with tf.Session():
        # Unpack the snapshot
        snapshot = joblib.load(pkl_filename)
        env = snapshot['env']
        policy = snapshot['policy']
        inference_net = snapshot['inference']

        task_envs = env.env._task_envs
        num_tasks = len(task_envs)
        goals = np.array([te._goal for te in task_envs])

        env = task_envs[0]
        env._goal = np.array([1., 1.])
        demonstrations = np.array(joblib.load('demonstrations.pkl'))
        obs = rollout_imitation(
                    TfEnv(env),
                    policy,
                    demonstrations[0],
                    inference_net,
                    max_path_length=MAX_PATH_LENGTH,
                    animated=True,)
        return obs



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s PKL_FILENAME' % sys.argv[0])
        sys.exit(0)

    imitate(sys.argv[1])
