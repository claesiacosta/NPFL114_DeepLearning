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

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")

def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)

    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    S = env.observation_space.n
    print(env.observation_space)
    A = env.action_space.n
    Q = np.zeros([S, A])
    C = np.zeros([S, A])
    for _ in range(args.episodes):
        # Perform episode, collecting states, actions and rewards
        states, actions, rewards = [], [], []
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of args.epsilon, use a random action,
            # otherwise choose and action with maximum Q[state, action].
            if np.random.uniform() < args.epsilon:
                action = np.random.randint(0, A)
            else:
                action = np.argmax(Q[state])
            print(action)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # TODO: Compute returns from the received rewards
        # and update Q and C.
        G = 0
        for t in reversed(range(len(rewards))):
            G += rewards[t-1]
            C[states[t], actions[t]] += 1
            Q[states[t], actions[t]] += (1 / C[states[t], actions[t]]) * (G - Q[states[t], actions[t]])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose greedy action
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed)

    main(env, args)
