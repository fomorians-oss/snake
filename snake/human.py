import argparse

import gym
import numpy as np
import tensorflow as tf
from gym.envs.registration import register


def main(args):
    # Random seeding.
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Make environment.
    register(
        id="Snake-v0",
        entry_point="snake.env:SnakeEnv",
        kwargs={"side_length": args.side_length},
    )
    env = gym.make("Snake-v0")
    state = env.reset()
    env.render()

    actions = ["l", "r", "u", "d"]
    actions_map = {action: i for i, action in enumerate(actions)}

    # Human control.
    while True:
        while True:
            action = input("> ")
            if action in actions:
                break
            else:
                print(f"Enter an action in {actions}")

        state, reward, done, _ = env.step(actions_map[action])
        env.render()

        if reward == 1:
            print("Fruit get :)")
        
        if done:
            state = env.reset()
            env.render()
            print("Snake death / max steps exceeded :(")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
