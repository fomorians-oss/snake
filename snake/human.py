import argparse

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.ion()


def main(args):
    # Random seeding.
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Make environment.
    env = gym.make("Snake-v0")
    state = env.reset()

    fig, ax = plt.subplots()
    im = ax.matshow(np.squeeze(state))
    plt.pause(1e-4)

    action_map = {"l": 0, "r": 1, "u": 2, "d": 3}

    # Human control.
    while True:
        while True:
            action = input("> ")
            if action in ["l", "r", "u", "d"]:
                break

        state, reward, done, _ = env.step(action_map[action])

        if done:
            state = env.reset()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
