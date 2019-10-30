import os
import time
import atexit
import argparse

import gym
import numpy as np
import tensorflow as tf
import pyoneer as pynr
import pyoneer.rl as pyrl

from snake import agents
from snake.rollout import BatchRollout
from snake.utilities import generalized_advantage_estimate, discounted_returns


def main(args):
    # Make job dir.
    timestamp = int(time.time())
    job_dir = os.path.join(args.job_dir, str(timestamp))
    os.makedirs(job_dir, exist_ok=True)

    # Make env.
    env = pyrl.wrappers.Batch(lambda: gym.make("Snake-v0"), batch_size=args.batch_size)
    atexit.register(env.close)

    # Random seeding.
    env.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Make agent.
    if args.agent == "mlp":
        agent = agents.MLPAgent(
            observation_space=env.observation_space, action_space=env.action_space
        )
    elif args.agent == "conv":
        agent = agents.ConvAgent(
            observation_space=env.observation_space, action_space=env.action_space
        )

    # Make optimizer.
    value_optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, clipnorm=1.0
    )
    policy_optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, clipnorm=1.0
    )

    # Make job for managing checkpoints.
    checkpointables = {
        "agent": agent,
        "value_optimizer": value_optimizer,
        "policy_optimizer": policy_optimizer,
    }
    checkpoint = tf.train.Checkpoint(**checkpointables)
    checkpoint_path = tf.train.latest_checkpoint(job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path)

    # Make summary writer
    summary_writer = tf.summary.create_file_writer(
        job_dir, max_queue=100, flush_millis=30 * 1000
    )
    summary_writer.set_as_default()

    # Make rollout.
    rollout = BatchRollout(env, max_episode_steps=args.max_steps)

    # Training loop.
    for it in range(args.n_iter):
        # Collect trajectories of experience.
        states, actions, rewards, next_states, weights, dones = rollout(
            agent=agent, episodes=args.batch_size, render=args.render, explore=True
        )
        episodic_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        lengths = tf.reduce_sum(weights, axis=1)
        mean_length = tf.reduce_mean(lengths)

        # Make summaries.
        tf.summary.scalar("rewards/train", episodic_reward, step=it)
        tf.summary.scalar("avg. length/train", mean_length, step=it)
        tf.summary.histogram("actions/train", actions, step=it)

        # Fit value network.
        for i in range(args.value_steps):
            # Get bootstrap values if training on one-step rollouts.
            next_value = agent.value(next_states)

            # Compute discounted returns to use as value network regression targets.
            returns = discounted_returns(
                rewards=tf.cast(rewards, tf.float32) * weights,
                discounts=args.discount,
                weights=weights,
            )

            with tf.GradientTape() as tape:
                # Compute value of states in sampled trajectories.
                values = agent.value(states)

                # Compute value loss as mean squared error
                # between predicted values and actual returns.
                value_loss = 0.5 * tf.reduce_sum(
                    (tf.square(values - tf.stop_gradient(returns)) * weights)
                )
                loss = value_loss / (args.batch_size * args.max_steps)

            # Compute and apply value network gradients.
            variables = agent.value_trainable_variables
            grads = tape.gradient(loss, variables)
            grad_norm = tf.linalg.global_norm(grads)
            value_optimizer.apply_gradients(zip(grads, variables))

            # Make summaries.
            tf.summary.histogram("values", values, step=value_optimizer.iterations)
            tf.summary.scalar("losses/critic", loss, step=value_optimizer.iterations)
            tf.summary.scalar(
                "grad_norm/critic", grad_norm, step=value_optimizer.iterations
            )

        # Fit policy network.
        for i in range(args.policy_steps):
            # Compute value baseline.
            values = agent.value(states)

            # Get bootstrap values if training on one-step rollouts.
            next_value = agent.value(next_states)

            # Compute advantages using TD(lambda).
            advantages = generalized_advantage_estimate(
                rewards=tf.cast(rewards, tf.float32),
                values=values,
                discounts=args.discount,
                lambdas=args.lmbda,
                weights=weights,
            )

            with tf.GradientTape() as tape:
                # Compute estimate of policy distribution.
                log_probs, values, probs = agent.policy_value(states, actions)

                # Compute unnormalized distribution implied by scaled, exponentiated advantages.
                score = tf.minimum(tf.exp(advantages / args.beta), args.score_max)

                # Compute policy loss as mismatch between policy and scaled advantage distribution.
                policy_loss = -tf.reduce_sum(
                    tf.squeeze(log_probs) * tf.stop_gradient(score) * weights
                )
                loss = policy_loss / (args.batch_size * args.max_steps)

            # Compute and apply gradients to policy parameters.
            variables = agent.policy_trainable_variables
            grads = tape.gradient(loss, variables)
            grad_norm = tf.linalg.global_norm(grads)
            policy_optimizer.apply_gradients(zip(grads, variables))

            entropy = -tf.reduce_mean(log_probs)
            value = tf.reduce_mean(values)

            # Make summaries.
            tf.summary.histogram(
                "policy/advantages", advantages, step=policy_optimizer.iterations
            )
            tf.summary.histogram(
                "policy/score", score, step=policy_optimizer.iterations
            )
            tf.summary.scalar(
                "losses/policy", policy_loss, step=policy_optimizer.iterations
            )
            tf.summary.scalar(
                "grad_norm/policy", grad_norm, step=policy_optimizer.iterations
            )
            tf.summary.scalar(
                "policy/entropy", entropy, step=policy_optimizer.iterations
            )
            tf.summary.scalar("policy/value", value, step=policy_optimizer.iterations)

            for j in range(env.action_space.n):
                tf.summary.histogram(
                    "action_probs/%d" % j,
                    probs[..., j],
                    step=policy_optimizer.iterations,
                )

        if it % args.valid_every == 0:
            # Collect trajectories of experience.
            _, actions, rewards, _, weights, _ = rollout(
                agent=agent, episodes=args.batch_size, render=args.render, explore=False
            )
            episodic_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
            lengths = tf.reduce_sum(weights, axis=1)
            mean_length = tf.reduce_mean(lengths)

            # Make summaries.
            tf.summary.scalar("rewards/eval", episodic_reward, step=it)
            tf.summary.scalar("avg. length/eval", mean_length, step=it)
            tf.summary.histogram("actions/eval", actions, step=it)

            # save checkpoint
            checkpoint_prefix = os.path.join(job_dir, "checkpoint")
            checkpoint.save(file_prefix=checkpoint_prefix)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", type=str, default="mlp")
    parser.add_argument("--n-iter", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--value-steps", type=int, default=10)
    parser.add_argument("--policy-steps", type=int, default=10)
    parser.add_argument("--valid-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--score-max", type=float, default=100.0)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
