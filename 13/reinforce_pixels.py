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

import cart_pole_pixels_environment
import wrappers
import zipfile

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=9, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--learning_rate", default=0.008, type=float, help="Learning rate.")
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--episodes", default=800, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=29, type=int, help="Size of hidden layer.")

class Agent:
    def __init__(self, env, args):
        inputs = tf.keras.layers.Input([80, 80, 3])

        hidden = tf.keras.layers.Conv2D(4, kernel_size= 3, strides=3, padding = 'same')(inputs)
        hidden = tf.keras.layers.Conv2D(4, kernel_size= 3, strides=3, padding = 'same')(hidden)
        hidden = tf.keras.layers.MaxPool2D(pool_size=3, strides=3)(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(units=args.hidden_layer_size, activation=tf.nn.relu)(hidden)
        outputs = tf.keras.layers.Dense(units=env.action_space.n, activation=tf.nn.softmax)(hidden)

        hidden_base = tf.keras.layers.Conv2D(4, kernel_size= 3, strides=3, padding = 'same')(inputs)
        hidden_base = tf.keras.layers.Conv2D(4, kernel_size= 3, strides=3, padding = 'same')(hidden_base)
        hidden_base = tf.keras.layers.MaxPool2D(pool_size=3, strides=3)(hidden_base)
        hidden_base = tf.keras.layers.Flatten()(hidden_base)
        hidden_base = tf.keras.layers.Dense(units=args.hidden_layer_size, activation=tf.nn.relu)(hidden_base)
        #hidden_base = tf.keras.layers.Dropout(0.2)(hidden_base)
        #hidden_base = tf.keras.layers.Dense(units=args.hidden_layer_size, activation=tf.nn.relu)(hidden_base)
        #hidden_base = tf.keras.layers.Dropout(0.2)(hidden_base)
        outputs_base = tf.keras.layers.Dense(1)(hidden_base)[:,0]

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

    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):

        with tf.GradientTape() as b_tape:
            predicted_baseline = self._model_baseline(states, training=True)
            b_loss = self._model_baseline.loss(returns, predicted_baseline)

        with tf.GradientTape() as p_tape:
            predictions = self._model(states, training=True)
            p_loss = self._model.loss(actions, predictions, sample_weight=returns - predicted_baseline)

        self._model_baseline.optimizer.minimize(b_loss, self._model_baseline.trainable_variables, tape=b_tape)

        self._model.optimizer.minimize(p_loss, self._model.trainable_variables, tape=p_tape)

        #self.optimizer.minimize(loss, [self._model.trainable_variables, self._model_baseline.trainable_variables], tape=tape)
        #return {"loss": loss}

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        #print(len(states))
        return self._model(states)
        #return self._model.predict_on_batch(x=states)

def test(env, args):
    agent = Agent(env, args)
    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall("model_p")
    agent._model = tf.keras.models.load_model('model_p/model/model')
    agent._model_baseline = tf.keras.models.load_model('model_p/model/model_baseline')

    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose an action
            probabilities = agent.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)

def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if args.recodex:
        # TODO: Perform evaluation of a trained model.
        test(env, args)
        
    else:
        agent = Agent(env, args)
        for _ in range(args.episodes // args.batch_size):
            batch_states, batch_actions, batch_returns = [], [], []
            # Perform arg.batch_size episodes ...
            for _ in range(args.batch_size):
                states, actions, rewards = [], [], []
                state, done = env.reset(), False

                while not done:
                    if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                        env.render()
                    probabilities = agent.predict([state])[0]
                    action = np.random.choice(a=env.action_space.n, p=probabilities)

                    next_state, reward, done, _ = env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state
                G = 0
                returns = []
                for t in reversed(range(len(rewards))):
                    G += rewards[t - 1]
                    returns.append(G)
                returns.reverse()

                batch_states += states
                batch_actions += actions
                batch_returns += returns
            
            agent.train(batch_states, batch_actions, batch_returns)

        agent._model_baseline.save('./model/model_baseline')
        agent._model.save('./model/model')

        while True:
            state, done = env.reset(start_evaluation=True), False
            while not done:
                action = np.argmax(agent.predict([state])[0])
                state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPolePixels-v0"), args.seed)

    main(env, args)
