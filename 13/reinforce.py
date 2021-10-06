### Team members:
## Jaroslava Schovancova
# a79d74c9-f42b-11e9-9ce9-00505601122b
## Antonia Claésia da Costa Souza
# 80acc802-7cec-11eb-a1a9-005056ad4f31
## Richard Hajek
# 33179fd1-18a0-11eb-8e81-005056ad4f31
###

# !/usr/bin/env python3
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate.")


class Agent:
    def __init__(self, env, args):
        # TODO: Create a suitable model. The predict method assumes
        # it is stored as `self._model`.
        #
        # Using Adam optimizer with given `args.learning_rate` is a good default.
        inputs = tf.keras.layers.Input(shape=env.observation_space.shape)
        hidden = tf.keras.layers.Dense(units=args.hidden_layer_size, activation=tf.nn.leaky_relu)(inputs)
        hidden = tf.keras.layers.Dense(units=args.hidden_layer_size, activation=tf.nn.leaky_relu)(hidden)
        outputs = tf.keras.layers.Dense(units=env.action_space.n, activation=tf.nn.softmax)(hidden)
        self._model = tf.keras.Model(inputs, outputs)
        self._model.compile(
            optimizer=tf.optimizers.Adam(lr=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        )

    # Define a training method.
    #
    # Note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):
        # TODO: Perform training, using the loss from the REINFORCE algorithm.
        # The easiest approach is to use the `sample_weight` argument of
        # tf.losses.Loss.__call__, but you can also construct the Loss object
        # with tf.losses.Reduction.NONE and perform the weighting manually.
        # print(states.shape)
        # states, actions, returns = np.array(states), np.array(actions), np.array(returns)

        with tf.GradientTape() as tape:
            predictions = self._model(states, actions, returns)
            loss = self._model.loss(actions, predictions, sample_weight=returns)

        self._model.optimizer.minimize(loss, self._model.trainable_variables, tape=tape)  # perform optimizer step

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self._model(states)


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the agent
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # TODO: Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                probabilities = agent.predict([state])[0]
                action = np.random.choice(len(probabilities), p=probabilities)

                next_state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO: Compute returns from the received rewards
            G = 0
            returns = []
            for t in reversed(range(len(rewards))):
                G += rewards[t - 1]
                returns.append(G)
            # TODO: Add states, actions and returns to the training batch
            returns.reverse()
            # print(returns)
            batch_states += states
            batch_actions += actions
            batch_returns += returns
        # TODO: Train using the generated batch.
        agent.train(batch_states, batch_actions, batch_returns)
    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose greedy action
            probabilities = agent.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)

    main(env, args)
