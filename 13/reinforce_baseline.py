### Team members:
## Jaroslava Schovancova
# a79d74c9-f42b-11e9-9ce9-00505601122b
## Antonia Claésia da Costa Souza
# 80acc802-7cec-11eb-a1a9-005056ad4f31
## Richard Hajek
# 33179fd1-18a0-11eb-8e81-005056ad4f31
###

#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

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
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=27, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.008, type=float, help="Learning rate.")

class Agent:
    def __init__(self, env, args):
        # TODO: Create a suitable model. The predict method assumes
        # the policy network is stored as `self._model`.
        #
        # Apart from the model defined in `reinforce`, define also another
        # model for computing the baseline (with one output, using a dense layer
        # without activation). (Alternatively, this baseline computation can
        # be grouped together with the policy computation in a single tf.keras.Model.)
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.
        inputs = tf.keras.layers.Input(shape=env.observation_space.shape)
        hidden = tf.keras.layers.Dense(units=args.hidden_layer_size, activation="tanh", kernel_regularizer='l2')(inputs)
        hidden = tf.keras.layers.Dropout(0.2)(hidden)
        hidden = tf.keras.layers.Dense(units=args.hidden_layer_size, activation="tanh", kernel_regularizer='l2')(hidden)
        hidden = tf.keras.layers.Dropout(0.2)(hidden)

        hidden_base = tf.keras.layers.Dense(units=args.hidden_layer_size, activation="tanh", kernel_regularizer='l2')(inputs)
        hidden_base = tf.keras.layers.Dropout(0.2)(hidden_base)
        hidden_base = tf.keras.layers.Dense(units=args.hidden_layer_size, activation="tanh", kernel_regularizer='l2')(hidden_base)
        hidden_base = tf.keras.layers.Dropout(0.2)(hidden_base)

        outputs = tf.keras.layers.Dense(units=env.action_space.n, activation=tf.nn.softmax)(hidden)

        outputs_base = tf.keras.layers.Dense(1)(hidden_base)

        self._model = tf.keras.Model(inputs, outputs)
        self._model.compile(
            optimizer=tf.optimizers.Adam(lr=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy()
        )

        self._model_baseline = tf.keras.Model(inputs, outputs_base)
        self._model_baseline.compile(
            optimizer=tf.optimizers.Adam(lr=args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError()
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
        # TODO: Perform training, using the loss from the REINFORCE with
        # baseline algorithm.
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the baseline model to predict `returns`

        with tf.GradientTape() as b_tape:
            predicted_baseline = self._model_baseline(states, training=True)
            b_loss = self._model_baseline.loss(returns, predicted_baseline)
        
		# - train the policy model, using `returns - predicted_baseline` as
        #   the advantage estimate
        with tf.GradientTape() as p_tape:
            predictions = self._model(states, training=True)
            p_loss = self._model.loss(actions, predictions, sample_weight=returns - tf.reshape(predicted_baseline, [-1]))

        self._model_baseline.optimizer.minimize(b_loss, self._model_baseline.trainable_variables, tape=b_tape)

        self._model.optimizer.minimize(p_loss, self._model.trainable_variables, tape=p_tape)

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

                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you

                # can compute using `agent.predict` and current `state`.
                print(state.shape)
                break
                probabilities = agent.predict([state])[0]
                action = np.random.choice(len(probabilities), p=probabilities)
                next_state, reward, done, _ = env.step(action)
                #print(reward)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute returns from the received rewards

            # TODO(reinforce): Add states, actions and returns to the training batch
            G = 0
            returns = []
            #print(actions.count(1))
            #print(actions.count(0))
            for t in reversed(range(len(rewards))):
                G += rewards[t - 1]
                returns.append(G)
            #returns.append(0)
            returns.reverse()

            batch_states += states
            batch_actions += actions
            batch_returns += returns
        # TODO(reinforce): Train using the generated batch.
        agent.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO(reinforce): Choose greedy action
            probabilities = agent.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)

    main(env, args)
