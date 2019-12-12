import numpy as np
import tensorflow as tf


class BatchRollout:
    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, agent, episodes, render=False, explore=True):
        assert len(self.env) == episodes

        observation_space = self.env.observation_space
        action_space = self.env.action_space

        states = np.zeros(
            shape=(episodes, self.max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        actions = np.zeros(
            shape=(episodes, self.max_episode_steps) + action_space.shape,
            dtype=action_space.dtype,
        )
        next_states = np.zeros(
            shape=(episodes, self.max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        rewards = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)
        weights = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)
        dones = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)

        batch_size = len(self.env)
        episode_done = np.zeros(shape=batch_size, dtype=np.bool)

        state = self.env.reset()

        for step in range(self.max_episode_steps):
            if render:
                self.env.envs[0].render()

            action = agent.step(
                tf.expand_dims(state, axis=1), explore=explore, reset_states=step == 0
            ).numpy()[:, 0]
            next_state, reward, done, info = self.env.step(action)

            states[:, step] = state
            actions[:, step] = action
            rewards[:, step] = reward
            next_states[:, step] = next_state
            weights[:, step] = np.where(episode_done, 0.0, 1.0)
            dones[:, step] = done

            # update episode done status
            episode_done = episode_done | done

            # end the rollout if all episodes are done
            if np.all(episode_done):
                break

            state = next_state

        # ensure rewards are masked
        rewards *= weights

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        weights = tf.convert_to_tensor(weights)
        dones = tf.convert_to_tensor(dones)

        return states, actions, rewards, next_states, weights, dones
