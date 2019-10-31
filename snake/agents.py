import numpy as np
import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


class MLPAgent(tf.Module):
    def __init__(self, observation_space, action_space):
        super(MLPAgent, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        # Weights initializers.
        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        # Hidden layers.
        self._dense1 = tf.keras.layers.Dense(
            units=128, activation=tf.nn.relu, kernel_initializer=kernel_initializer
        )
        self._dense2 = tf.keras.layers.Dense(
            units=64, activation=tf.nn.relu, kernel_initializer=kernel_initializer
        )

        # Output layers (logits for policy, value head).
        self._logits = tf.keras.layers.Dense(
            units=action_space.n, kernel_initializer=logits_initializer
        )
        self._value = tf.keras.layers.Dense(1)

    @property
    def value_trainable_variables(self):
        return (
            self._dense1.trainable_variables
            + self._dense2.trainable_variables
            + self._value.trainable_variables
        )

    @property
    def policy_trainable_variables(self):
        return (
            self._dense1.trainable_variables
            + self._dense2.trainable_variables
            + self._logits.trainable_variables
        )

    def _scale_state(self, state):
        state = tf.cast(state, dtype=tf.float32)
        observation_high = np.where(
            self.observation_space.high < np.finfo(np.float32).max,
            self.observation_space.high,
            +1.0,
        )
        observation_low = np.where(
            self.observation_space.low > np.finfo(np.float32).min,
            self.observation_space.low,
            -1.0,
        )

        observation_mean, observation_var = pynr.moments.range_moments(
            observation_low, observation_high
        )
        state_norm = tf.math.divide_no_nan(
            state - observation_mean, tf.sqrt(observation_var)
        )
        return state_norm

    @tf.function
    def _hidden(self, state):
        state = self._scale_state(state)
        batch_size, time_steps = tf.shape(state)[0], tf.shape(state)[1]
        state = tf.reshape(state, [batch_size, time_steps, -1])
        return self._dense2(self._dense1(state))

    @tf.function
    def value(self, state):
        hidden = self._hidden(state)
        value = tf.squeeze(self._value(hidden), axis=-1)
        return value

    def policy(self, state):
        hidden = self._hidden(state)
        logits = self._logits(hidden)
        policy = tfp.distributions.Categorical(logits=logits)
        return policy

    @tf.function
    def policy_value(self, state, action):
        policy = self.policy(state)
        values = self.value(state)

        log_probs = policy.log_prob(action)
        probs = policy.probs_parameter()

        return log_probs, values, probs

    @tf.function
    def step(self, state, explore=True):
        policy = self.policy(state)

        if explore:
            action = policy.sample()
        else:
            action = policy.mode()

        return action


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.filters = filters

        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=None,
            kernel_initializer=kernel_initializer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=None,
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs):
        hidden = pynr.activations.swish(inputs)
        hidden = self.conv1(hidden)
        hidden = pynr.activations.swish(hidden)
        hidden = self.conv2(hidden)
        hidden += inputs
        return hidden


class ConvAgent(tf.Module):
    def __init__(self, observation_space, action_space):
        super(ConvAgent, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        # Weights initializers.
        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        # Hidden layers.
        self._conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=2,
            strides=1,
            padding="same",
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )

        self._downsample1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self._residual_block1 = ResidualBlock(filters=16)

        self._global_pool = tf.keras.layers.GlobalMaxPool2D()

        self._dense_hidden = tf.keras.layers.Dense(
            units=64,
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )

        # Output layers (logits for policy, value head).
        self._logits = tf.keras.layers.Dense(
            units=action_space.n, kernel_initializer=logits_initializer
        )
        self._value = tf.keras.layers.Dense(1)

    @property
    def value_trainable_variables(self):
        return (
            self._conv1.trainable_variables
            + self._residual_block1.trainable_variables
            + self._dense_hidden.trainable_variables
            + self._value.trainable_variables
        )

    @property
    def policy_trainable_variables(self):
        return (
            self._conv1.trainable_variables
            + self._residual_block1.trainable_variables
            + self._dense_hidden.trainable_variables
            + self._logits.trainable_variables
        )

    def _scale_state(self, state):
        state = tf.cast(state, dtype=tf.float32)
        observation_high = np.where(
            self.observation_space.high < np.finfo(np.float32).max,
            self.observation_space.high,
            +1.0,
        )
        observation_low = np.where(
            self.observation_space.low > np.finfo(np.float32).min,
            self.observation_space.low,
            -1.0,
        )

        observation_mean, observation_var = pynr.moments.range_moments(
            observation_low, observation_high
        )
        state_norm = tf.math.divide_no_nan(
            state - observation_mean, tf.sqrt(observation_var)
        )
        return state_norm

    @tf.function
    def _hidden(self, state):
        state = self._scale_state(state)

        input_shape = tf.shape(state)
        batch_size = input_shape[0]
        time = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]

        state = tf.reshape(state, [batch_size * time, height, width, channels])

        hidden = self._conv1(state)
        hidden = self._downsample1(hidden)
        hidden = self._residual_block1(hidden)
        hidden = tf.reshape(hidden, [batch_size, time, -1])
        hidden = self._dense_hidden(hidden)
        return hidden

    @tf.function
    def value(self, state):
        hidden = self._hidden(state)
        value = tf.squeeze(self._value(hidden), axis=-1)
        return value

    def policy(self, state):
        hidden = self._hidden(state)
        logits = self._logits(hidden)
        policy = tfp.distributions.Categorical(logits=logits)
        return policy

    @tf.function
    def policy_value(self, state, action):
        policy = self.policy(state)
        values = self.value(state)

        log_probs = policy.log_prob(action)
        probs = policy.probs_parameter()

        return log_probs, values, probs

    @tf.function
    def step(self, state, explore=True):
        policy = self.policy(state)

        if explore:
            action = policy.sample()
        else:
            action = policy.mode()

        return action
