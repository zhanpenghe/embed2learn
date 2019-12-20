
import sys
import time

import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from embed2learn.envs.util import colormap_mpl
from embed2learn.envs.multi_task_env import normalize, TfEnv


def play(pkl_filename):
    with tf.Session():
        # Unpack the snapshot
        snapshot = joblib.load(pkl_filename)
        env = snapshot['env']
        policy = snapshot['policy']

        data = []
        for _ in range(10):
            obs = env.reset()
            data.append([obs])
            for _ in range(50):
                env.render()
                a, _ = policy.get_action(obs)
                obs, _, _, _ = env.step(a)
                data[-1].append(obs)
        joblib.dump(data, 'demonstrations.pkl')
        import ipdb
        ipdb.set_trace()



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s PKL_FILENAME' % sys.argv[0])
        sys.exit(0)

    play(sys.argv[1])
