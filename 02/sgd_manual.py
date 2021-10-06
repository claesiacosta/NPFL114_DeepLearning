# Ríša
# 33179fd1-18a0-11eb-8e81-005056ad4f31
# Jaroslava
# a79d74c9-f42b-11e9-9ce9-00505601122b
# Claésia
# 80acc802-7cec-11eb-a1a9-005056ad4f31

#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Model(tf.Module):
    def __init__(self, args):
        self._args = args

        self._W1 = tf.Variable(tf.random.normal([MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed), trainable=True)
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)

        # TODO(sgd_backpropagation): Create variables:
        # - _W2, which is a trainable Variable of size [args.hidden_layer, MNIST.LABELS],
        #   initialized to `tf.random.normal` value with stddev=0.1 and seed=args.seed,
        self._W2 = tf.Variable(tf.random.normal([args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed), trainable=True)
        # - _b2, which is a trainable Variable of size [MNIST.LABELS] initialized to zeros
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]), trainable=True)


    def predict(self, inputs):
        # TODO(sgd_backpropagation): Define the computation of the network. Notably:
        # - start by reshaping the inputs to shape [inputs.shape[0], -1].
        #   The -1 is a wildcard which is computed so that the number
        #   of elements before and afte the reshape fits.
        input_reshaped = res = tf.reshape(inputs, [inputs.shape[0], -1])
        # - then multiply the inputs by `self._W1` and then add `self._b1`
        res = res @ self._W1 + self._b1
        # - apply `tf.nn.tanh`
        hidden = res = tf.nn.tanh(res)
        # - multiply the result by `self._W2` and then add `self._b2`
        res = res @ self._W2 + self._b2
        # - finally apply `tf.nn.softmax` and return the result
        res = tf.nn.softmax(res)
        # TODO: In order to support manual gradient computation, you should
        # return not only the output layer, but also the hidden layer after applying
        # tf.nn.tanh, and the input layer after reshaping.
        return (res, hidden, input_reshaped)

    def train_epoch(self, dataset):
        for batch in dataset.batches(self._args.batch_size):
            # The batch contains
            # - batch["images"] with shape [?, MNIST.H, MNIST.W, MNIST.C]
            # - batch["labels"] with shape [?]
            # Size of the batch is `self._args.batch_size`, except for the last, which
            # might be smaller.

            # TODO: Contrary to sgd_backpropagation, the goal here is to compute
            # the gradient manually, without tf.GradientTape. ReCodEx checks
            # that `tf.GradientTape` is not used and if it is, your solution does
            # not pass.

            # TODO: Compute the input layer, hidden layer and output layer
            # of the batch images using `self.predict`.
            probabilities, hidden, input_reshaped = self.predict(inputs=batch['images'])

            # TODO: Compute the gradient of the loss with respect to all
            # variables. Note that the loss is computed as:
            # - for every batch example, it is the categorical crossentropy of the
            #   predicted probabilities and gold batch label
            # - finally, the individual batch example losses are averaged

            # During the gradient computation, you will need to compute
            # a so-called outer product
            #   `C[a, i, j] = A[a, i] * B[a, j]`
            # which you can for example as
            #   `A[:, :, tf.newaxis] * B[:, tf.newaxis, :]`
            # or with
            #   `tf.einsum("ai,aj->aij", A, B)`

            x_softmax = probabilities - tf.one_hot(batch["labels"], MNIST.LABELS)
            grad_b2 = tf.math.reduce_mean(x_softmax, axis=0)

            grad_w2 = tf.einsum("ai,aj->aij", hidden, x_softmax)
            grad_w2 = tf.math.reduce_mean(grad_w2, axis=0)

            tanh = tf.math.reciprocal(tf.math.square(tf.cosh(input_reshaped @ self._W1 + self._b1)))
            x_softmax_w2 = tf.multiply(x_softmax @ tf.transpose(self._W2), tanh)
            grad_b1 = tf.math.reduce_mean(x_softmax_w2, axis=0)

            grad_w1 = tf.einsum("ai,aj->aij", input_reshaped, x_softmax_w2)
            grad_w1 = tf.math.reduce_mean(grad_w1, axis=0)

            # TODO(sgd_backpropagation): Perform the SGD update with learning rate `self._args.learning_rate`
            # for the variable and computed gradient. You can modify
            # variable value with `variable.assign` or in this case the more
            # efficient `variable.assign_sub`.
            variables = [self._W1, self._b1, self._W2, self._b2]
            gradients = [grad_w1, grad_b1, grad_w2, grad_b2]

            for variable, gradient in zip(variables, gradients):
                # TODO: Perform the SGD update with learning rate `self._args.learning_rate`
                # for the variable and computed gradient. You can modify
                # variable value with `variable.assign` or in this case the more
                # efficient `variable.assign_sub`.
                variable.assign_sub(self._args.learning_rate * gradient)

    def evaluate(self, dataset):
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(self._args.batch_size):
            # TODO(sgd_backpropagation): Compute the probabilities of the batch images
#            probabilities = ...
            probabilities, hidden, input_reshaped = self.predict(inputs=batch['images'])

            # TODO(sgd_backpropagation): Evaluate how many batch examples were predicted
            # correctly and increase `correct` variable accordingly.
            correct += np.sum(batch["labels"] == np.argmax(probabilities, axis=1))

        return correct / dataset.size


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10*1000)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        # TODO(sgd_backpropagation): Run the `train_epoch` with `mnist.train` dataset
        model.train_epoch(dataset = mnist.train)

        # TODO(sgd_backpropagation): Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
        with writer.as_default(step=epoch + 1):
            tf.summary.scalar("dev/accuracy", 100 * accuracy)

    # TODO(sgd_backpropagation): Evaluate the test data using `evaluate` on `mnist.test` dataset
    accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
    with writer.as_default(step=epoch + 1):
        tf.summary.scalar("test/accuracy", 100 * accuracy)

    # Return test accuracy for ReCodEx to validate
    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
