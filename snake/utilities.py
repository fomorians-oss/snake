import tensorflow as tf


@tf.function
def temporal_difference(returns, values, back_prop=False, name=None):
    """Computes the temporal difference.
    Args:
        returns: tensor of shape [Batch x Time], [Batch x Time x ...]
        values: tensor of shape [Batch x Time], [Batch x Time x ...]
        back_prop: allow back_prop through the calculation.
        name: optional op name.
    Returns:
        tensor of shape [Batch x Time]
    """
    with tf.name_scope(name or "BaselineAdvantageEstimate"):
        # Check returns shape.
        returns = tf.convert_to_tensor(returns)
        assert returns.shape.rank >= 2, "returns must be atleast rank 2."

        # Check values shape.
        values = tf.convert_to_tensor(values)
        assert values.shape.rank >= 2, "values must be atleast rank 2."

        assert (
            returns.shape.rank <= values.shape.rank
        ), "values rank must be == returns rank."

        advantages = returns - values
        advantages = tf.debugging.check_numerics(advantages, "Advantages")
        if not back_prop:
            advantages = tf.stop_gradient(advantages)
        return advantages


@tf.function
def generalized_advantage_estimate(
    rewards,
    values,
    last_value=None,
    discounts=0.99,
    lambdas=0.975,
    weights=1.0,
    time_major=False,
    back_prop=False,
    name=None,
):
    """Computes Generalized Advantage Estimation.
    Args:
        rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
        values: tensor of shape [Batch x Time], [Batch x Time x ...].
        last_value: tensor of shape [], [Batch], [Batch x ...].
        discounts: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        lambdas: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        weights: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        time_major: flag if tensors are time_major.
            Batch and Time are transposed in this doc.
        back_prop: allow back_prop through the calculation.
        name: optional op name.
    Returns:
        tensor of shape [Batch x Time] or [Time x Batch]
    """
    with tf.name_scope(name or "GeneralizedAdvantageEstimate"):
        batch_axis = int(time_major)
        time_axis = int(not time_major)

        # Check rewards shape.
        rewards = tf.convert_to_tensor(rewards)
        rewards_rank = rewards.shape.rank
        assert rewards_rank >= 2, "rewards must be atleast rank 2."
        rewards_shape = tf.shape(rewards)

        # Check discounts shape, broadcast to rewards shape.
        discounts = tf.convert_to_tensor(discounts)
        assert (
            discounts.shape.rank <= rewards.shape.rank
        ), "discounts rank must be <= rewards rank."
        discounts = tf.broadcast_to(discounts, rewards_shape)

        # Check lambdas shape, broadcast to rewards shape.
        lambdas = tf.convert_to_tensor(lambdas)
        assert (
            lambdas.shape.rank <= rewards.shape.rank
        ), "lambdas rank must be <= rewards rank."
        lambdas = tf.broadcast_to(lambdas, rewards_shape)

        # Check weights shape, broadcast to weights shape.
        weights = tf.convert_to_tensor(weights)
        assert (
            weights.shape.rank <= rewards.shape.rank
        ), "weights rank must be <= rewards rank."
        weights = tf.broadcast_to(weights, rewards_shape)

        if last_value is None:
            if time_major:
                last_value = tf.zeros_like(values[-1, :])
            else:
                last_value = tf.zeros_like(values[:, -1])
        else:
            last_value = tf.convert_to_tensor(last_value)

        last_value_t = tf.expand_dims(last_value, axis=time_axis)

        # Shift values to tp1.
        if time_major:
            values_tp1 = values[1:, :]
        else:
            values_tp1 = values[:, 1:]

        # Pad the values with the last value.
        values_tp1 = tf.concat([values_tp1, last_value_t], axis=time_axis)

        delta = temporal_difference(
            rewards + discounts * values_tp1, values, back_prop=back_prop
        )

        def reduce_fn(agg, cur):
            next_agg = cur[0] + cur[1] * agg
            return next_agg

        if time_major:
            elements = (delta * weights, lambdas * discounts * weights)
        else:
            rank_list = list(range(2, rewards_rank))
            elements = (
                tf.transpose(delta * weights, [1, 0] + rank_list),
                tf.transpose(lambdas * discounts * weights, [1, 0] + rank_list),
            )

        advantages = tf.scan(
            fn=reduce_fn,
            elems=elements,
            initializer=tf.zeros_like(last_value),
            parallel_iterations=1,
            back_prop=back_prop,
            reverse=True,
        )

        if not time_major:
            rank_list = list(range(2, rewards_rank))
            advantages = tf.transpose(advantages, [1, 0] + rank_list)
        advantages = advantages * weights
        advantages = tf.debugging.check_numerics(advantages, "Advantages")
        if not back_prop:
            advantages = tf.stop_gradient(advantages)
        return advantages


@tf.function
def discounted_returns(
    rewards,
    bootstrap_value=None,
    discounts=0.99,
    weights=1.0,
    time_major=False,
    back_prop=False,
    name=None,
):
    """Compute discounted returns.
    Args:
        rewards: tensor of shape [Batch x Time], [Batch x Time x ...]
        bootstrap_value: The discounted n-step value. A tensor of shape [],
            [...], [Batch], [Batch, ...], [Batch x Time], [Batch x Time x ...].
        discounts: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        weights: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        time_major: flag if tensors are time_major.
            Batch and Time are transposed in this doc.
        back_prop: allow back_prop through the calculation.
        name: optional op name.
    Returns:
        tensor of shape [Batch x Time x ...] or [Time x Batch x ...]
    """
    with tf.name_scope(name or "DiscountedReturns"):
        batch_axis = int(time_major)

        # Check rewards shape.
        rewards = tf.convert_to_tensor(rewards)
        rewards_rank = rewards.shape.rank
        assert rewards_rank >= 2, "rewards must be atleast rank 2."
        rewards_shape = tf.shape(rewards)

        # Compute the batch shape of rewards.
        if time_major:
            reward_shape_no_time = rewards_shape[1:]
        else:
            reward_shape_no_time = rewards_shape[:1]
            if rewards_rank > 2:
                reward_shape_no_time = tf.concat(
                    [reward_shape_no_time, rewards_shape[2:]], axis=-1
                )

        # Check discounts shape, broadcast to rewards shape.
        discounts = tf.convert_to_tensor(discounts)
        assert (
            discounts.shape.rank <= rewards.shape.rank
        ), "discounts rank must be <= rewards rank."
        discounts = tf.broadcast_to(discounts, rewards_shape)

        # Check weights shape, broadcast to weights shape.
        weights = tf.convert_to_tensor(weights)
        assert (
            weights.shape.rank <= rewards.shape.rank
        ), "weights rank must be <= rewards rank."
        weights = tf.broadcast_to(weights, rewards_shape)

        # Check if bootstrap values are supplied. If bootstrap values exist,
        # we want them to be the same shape as the batch.
        if bootstrap_value is None:
            bootstrap_value = tf.zeros(reward_shape_no_time, rewards.dtype)
        else:
            bootstrap_value = tf.convert_to_tensor(bootstrap_value)
            bootstrap_value = tf.broadcast_to(bootstrap_value, reward_shape_no_time)

        def reduce_fn(agg, cur):
            next_agg = cur[0] + cur[1] * agg
            return next_agg

        if not time_major:
            rank_list = list(range(2, rewards_rank))
            rewards = tf.transpose(rewards, [1, 0] + rank_list)
            discounts = tf.transpose(discounts, [1, 0] + rank_list)

        returns = tf.scan(
            fn=reduce_fn,
            elems=[rewards, discounts],
            initializer=bootstrap_value,
            parallel_iterations=1,  # chronological.
            back_prop=back_prop,
            reverse=True,
        )

        if not time_major:
            rank_list = list(range(2, rewards_rank))
            returns = tf.transpose(returns, [1, 0] + rank_list)

        returns = returns * weights
        returns = tf.debugging.check_numerics(returns, "Returns")
        if not back_prop:
            returns = tf.stop_gradient(returns)
        return returns
