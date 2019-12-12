from collections import deque

import time

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.envs.classic_control.rendering import SimpleImageViewer


LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

FRUIT = 0
BODY = 1
HEAD = 2

DEFAULT_REWARD = 0
DEATH_REWARD = 0
FRUIT_REWARD = 1


def _repeat_axes(x, factor, axis=[0, 1]):
    """Repeat np.array tiling it by `factor` on all axes.

    Args:
        x: input array.
        factor: number of repeats per axis.
        axis: axes to repeat x by factor.
    Returns:
        repeated array with shape `[x.shape[ax] * factor for ax in axis]`
    """
    for ax in axis:
        x = np.repeat(x, factor, axis=ax)

    return x


class SnakeEnv(gym.Env):
    """Implements an OpenAI gym interface to the classic Snake video game."""

    def __init__(self, side_length=8, egocentric=False, egocentric_side_length=3):
        assert (
            egocentric_side_length % 2 == 1
        ), "Egocentric perspective view must have odd-length sides."

        self.side_length = side_length
        self.egocentric = egocentric
        self.egocentric_side_length = egocentric_side_length

        self.action_space = gym.spaces.Discrete(4)

        if egocentric:
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(egocentric_side_length, egocentric_side_length, 3),
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(side_length, side_length, 3)
            )

        self.viewer = None
        self.resize_scale = 64
        self.delay = 0.01

        self._max_episode_steps = 100

    def _get_obs(self):
        if self.egocentric:
            extent = self.egocentric_side_length // 2
            grid = self.grid.reshape(self.side_length, self.side_length, 3)
            grid = tf.pad(
                grid,
                pad_width=((extent, extent), (extent, extent), (0, 0)),
                constant_values=-1,
            )

            snake_row = self.snake_position // self.side_length + extent
            snake_column = self.snake_position % self.side_length + extent

            obs = grid[
                snake_row - extent : snake_row + extent + 1,
                snake_column - extent : snake_column + extent + 1,
            ]
            obs = tf.cast(obs, tf.float32)
        else:
            obs = tf.reshape(self.grid, [self.side_length, self.side_length, 3])
            obs = tf.cast(obs, tf.float32)

        return obs

    def _check_open_positions(self):
        return tf.cast(tf.where((self.grid[..., HEAD] + self.grid[..., BODY]) == 0)[:, 0], tf.int32)

    def _reset_grid_and_snake(self):
        self.grid = tf.zeros([self.side_length ** 2, 3])
        self.snake_position = tf.random.uniform(
            (), minval=0, maxval=self.side_length ** 2, dtype=tf.int32
        )
        self.grid = tf.tensor_scatter_nd_update(
            self.grid, [[self.snake_position, HEAD]], [1]
        )

    def _reset_fruit(self):
        self.grid = tf.transpose(
            tf.tensor_scatter_nd_update(
                tf.transpose(self.grid), [[FRUIT]], [tf.zeros([self.side_length ** 2])]
            )
        )
        open_positions = self._check_open_positions()
        n_open_positions = tf.shape(open_positions)[0]
        fruit_index = tf.random.uniform(
            (), minval=0, maxval=n_open_positions, dtype=tf.int32
        )
        self.fruit_position = open_positions[fruit_index]
        self.grid = tf.tensor_scatter_nd_update(
            self.grid, [[self.fruit_position, FRUIT]], [1]
        )
    
    def _get_fruit(self):
        self.length += 1
        self.grid = tf.tensor_scatter_nd_update(
            self.grid, [[self.queue[-1], BODY]], [1]
        )
        self.grid = tf.tensor_scatter_nd_update(
            self.grid, [[self.queue[-1], HEAD]], [0]
        )
        self.queue.append(self.snake_position)
        self.grid = tf.tensor_scatter_nd_update(
            self.grid, [[self.snake_position, HEAD]], [1]
        )

    def _no_fruit(self):
        self.grid = tf.tensor_scatter_nd_update(
            self.grid, [[self.queue[-1], BODY]], [1]
        )
        self.grid = tf.tensor_scatter_nd_update(
            self.grid, [[self.queue[-1], HEAD]], [0]
        )
        self.queue.append(self.snake_position)
        self.grid = tf.tensor_scatter_nd_update(
            self.grid, [[self.snake_position, HEAD]], [1]
        )

        last_position = self.queue.popleft()
        self.grid = tf.tensor_scatter_nd_update(
            self.grid, [[last_position, BODY]], [0]
        )

    def reset(self):
        self._reset_grid_and_snake()
        self._reset_fruit()

        self.length = 1
        self.queue = deque([self.snake_position])

        state = self._get_obs()
        return state

    def step(self, action):
        grid = self._get_obs()

        snake_row = self.snake_position // self.side_length
        snake_column = self.snake_position % self.side_length
        fruit_row = self.fruit_position // self.side_length
        fruit_column = self.fruit_position % self.side_length

        reward = DEFAULT_REWARD
        done = False

        if action == LEFT:
            next_snake_position = self.side_length * snake_row + snake_column - 1

            if snake_column == 0 or next_snake_position in self.queue:
                done = True
                reward = DEATH_REWARD
            else:
                if (snake_row, snake_column - 1) == (fruit_row, fruit_column):
                    reward = FRUIT_REWARD
                    self.snake_position = next_snake_position
                    self._get_fruit()

                    if len(self._check_open_positions()) == 0:
                        done = True
                    else:
                        self._reset_fruit()
                else:
                    self.snake_position = next_snake_position
                    self._no_fruit()

        elif action == RIGHT:
            next_snake_position = self.side_length * snake_row + snake_column + 1

            if (
                snake_column == self.side_length - 1
                or next_snake_position in self.queue
            ):
                done = True
                reward = DEATH_REWARD
            else:
                if (snake_row, snake_column + 1) == (fruit_row, fruit_column):
                    reward = FRUIT_REWARD
                    self.snake_position = next_snake_position
                    self._get_fruit()

                    if len(self._check_open_positions()) == 0:
                        done = True
                    else:
                        self._reset_fruit()
                else:
                    self.snake_position = next_snake_position
                    self._no_fruit()

        elif action == UP:
            next_snake_position = self.side_length * (snake_row - 1) + snake_column

            if snake_row == 0 or next_snake_position in self.queue:
                done = True
                reward = DEATH_REWARD
            else:
                if (snake_row - 1, snake_column) == (fruit_row, fruit_column):
                    reward = FRUIT_REWARD
                    self.snake_position = next_snake_position
                    self._get_fruit()

                    if len(self._check_open_positions()) == 0:
                        done = True
                    else:
                        self._reset_fruit()
                else:
                    self.snake_position = next_snake_position
                    self._no_fruit()


        elif action == DOWN:
            next_snake_position = self.side_length * (snake_row + 1) + snake_column

            if snake_row == self.side_length - 1 or next_snake_position in self.queue:
                done = True
                reward = DEATH_REWARD
            else:
                if (snake_row + 1, snake_column) == (fruit_row, fruit_column):
                    reward = FRUIT_REWARD
                    self.snake_position = next_snake_position
                    self._get_fruit()

                    if len(self._check_open_positions()) == 0:
                        done = True
                    else:
                        self._reset_fruit()
                else:
                    self.snake_position = next_snake_position
                    self._no_fruit()

        state = self._get_obs()
        return state, reward, done, {}

    def render(self, mode="human"):
        """Render the board to an image viewer.

        Args:
            mode: One of the following modes:
                - "human": render to an image viewer.
        Returns:
            3D np.array (np.uint8) or a `viewer.isopen`.
        """
        img = self._get_obs().numpy()
        img = np.abs(img)
        img = _repeat_axes(img, factor=50, axis=[0, 1])
        img *= 255

        img = img.astype(np.uint8)
        if mode == "human":
            if self.viewer is None:
                self.viewer = SimpleImageViewer()

            self.viewer.imshow(img)
            time.sleep(self.delay)
            return self.viewer.isopen
