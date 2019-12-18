import sys

import joblib
import tensorflow as tf


def play(pickle_path):
    with tf.Session() as sess:
        snapshot = joblib.load(pickle_path)
        print(snapshot.keys())
        import ipdb
        ipdb.set_trace()
        print('Loaded')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s PKL_FILENAME' % sys.argv[0])
        sys.exit(0)

    play(sys.argv[1])
