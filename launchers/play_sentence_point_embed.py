import sys
import time
import itertools
import collections

import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from embed2learn.envs.util import colormap_mpl
from embed2learn.envs.multi_task_env import normalize, TfEnv

MAX_PATH_LENGTH = 50
SAMPLING_POSITIONS = np.linspace(-1, 1, num=10)


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def description_to_code(description, dictionary, max_sentence_length):
    code = []
    for s in description.split(' '):
        code.append(dictionary[s])
    while len(code) < max_sentence_length:
        code.append(0)
    return code


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
        a, agent_info = agent.get_action_from_latent(o, z)
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
    return latent_info['mean'], latent_info['log_std']


def play(pkl_filename):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(8, 8))

    with tf.Session():
        # Unpack the snapshot
        snapshot = joblib.load(pkl_filename)
        env = snapshot["env"]
        policy = snapshot["policy"]

        # Get the task goals        
        task_envs = env.env._task_envs
        num_tasks = len(task_envs)
        task_cmap = colormap_mpl(num_tasks)

        goal_descriptions = ['move right', 'move up', 'move left', 'move down']
        words = list(itertools.chain.from_iterable([s.split(' ') for s in goal_descriptions]))
        _, num_words, dictionary, rev_dictionary = build_dataset(words, len(words))
        max_sentence_length = 4
        sentence_code_dim = len(num_words)
        goal_codes = [description_to_code(s, dictionary, max_sentence_length) for s in goal_descriptions]
        goals = np.array([e._goal for e in task_envs])

        for t, env in enumerate(task_envs):
            # Get latent distribution
            z_mean, z_std = get_z_dist(goal_codes[t], policy)
            # transform_z(policy)

            # Plot goal
            plt.scatter(
                [goals[t, 0]], [goals[t, 1]],
                s=50,
                color=task_cmap[t],
                zorder=2,
                label="Task {}".format(t + 1))

            # Plot rollouts for linearly interpolated latents
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

                # Plot rollout
                plt.plot(obs[:, 0], obs[:, 1], alpha=0.7, color=task_cmap[t])

        plt.grid(True)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.axes().set_aspect('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig('rollout.pdf')
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s PKL_FILENAME' % sys.argv[0])
        sys.exit(0)

    play(sys.argv[1])
    
