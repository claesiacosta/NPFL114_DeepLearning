# Ríša
# 33179fd1-18a0-11eb-8e81-005056ad4f31
# Claésia
# 80acc802-7cec-11eb-a1a9-005056ad4f31
# Jaroslava
# a79d74c9-f42b-11e9-9ce9-00505601122b


#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--test", default=False, action="store_true", help="whether to run unit tests.")



class Convolution:
    def __init__(self, channels, kernel_size, stride, input_shape):
        # Create convolutional layer with the given arguments
        # and given input shape (e.g., [28, 28, 1]).
        self._channels: int = channels
        self._kernel_size: int = kernel_size
        self._stride: int = stride

        # Here the kernel and bias variables are created
        self._kernel = tf.Variable(
            tf.initializers.GlorotUniform(seed=42)(
                [self._kernel_size, self._kernel_size, input_shape[2], self._channels]),
            trainable=True)
        self._bias = tf.Variable(tf.initializers.Zeros()([self._channels]), trainable=True)

    def forward_kernel(self, kernel, inputs):

        (batch_size, input_height, input_width, input_channels) = inputs.shape
        output_height = ((input_height - self._kernel_size) // self._stride) + 1
        output_width = ((input_width - self._kernel_size) // self._stride) + 1
        output = tf.Variable(tf.zeros((batch_size, output_height, output_width, self._channels)))

        for h in range(self._kernel_size):
            for w in range(self._kernel_size):
                sliced = inputs[::, h::self._stride, w::self._stride][::, :output_height:, :output_width:]
                output = output + sliced @ kernel[h, w]

        output += self._bias

        return tf.nn.relu(output)

    def forward(self, inputs):
        return self.forward_kernel(self._kernel, inputs)

    def backward(self, inputs, outputs, outputs_gradient):
        # TODO: Given the inputs of the layer, outputs of the layer
        # (computed in forward pass) and the gradient of the loss
        # with respect to layer outputs, return a list with the
        # following three elements:
        # - gradient of inputs with respect to the loss
        # - list of variables in the layer, e.g.,
        #     [self._kernel, self._bias]
        # - list of gradients of the layer variables with respect
        #   to the loss (in the same order as the previous argument
        #
        # hidden_gradient, convolution_variables, convolution_gradients

        (batch_size, input_height, input_width, input_channels) = inputs.shape
        output_height = ((input_height - self._kernel_size) // self._stride) + 1
        output_width = ((input_width - self._kernel_size) // self._stride) + 1

        hidden_gradient = tf.Variable(tf.zeros(shape=tf.shape(inputs)))
        kernel_gradient = tf.Variable(tf.zeros(shape=tf.shape(self._kernel)))

        derivations = tf.math.sign(tf.nn.relu(outputs)) # derivation of relu at the point of outputs
        gradient_relu = derivations * outputs_gradient # gradient with respect to the output derivation

        for h in range(self._kernel_size):
            for w in range(self._kernel_size):

                sliced = inputs[::, h::self._stride, w::self._stride][::, :output_height:, :output_width:]
                sliced = tf.transpose(sliced, perm=(0, 1, 3, 2))
                s = sliced @ gradient_relu
                s = tf.math.reduce_sum(s, axis=(0, 1))
                kernel_gradient[h, w].assign(kernel_gradient[h, w] + s)


                relevant_hidden_gradient = hidden_gradient[::,
                                           h:h + (output_height * self._stride):self._stride,
                                           w:w + (output_width * self._stride):self._stride
                                           ]

                transposed_hw_kernel = tf.transpose(self._kernel[h, w])

                relevant_hidden_gradient.assign(relevant_hidden_gradient + gradient_relu @ transposed_hw_kernel)


        bias_gradient = tf.reduce_sum(gradient_relu, axis=(0, 1, 2))

        # dx, dW, db = self.conv_backward(inputs, outputs, outputs_gradient)
        return hidden_gradient, [self._kernel, self._bias], [kernel_gradient, bias_gradient]


class Network:
    def __init__(self, args):
        self._args = args

        # Create the convolutional layers according to `args.cnn`.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convolutions = []
        for layer in args.cnn.split(","):
            channels, kernel_size, stride = map(int, layer.split("-"))
            self._convolutions.append(Convolution(channels, kernel_size, stride, input_shape))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, channels]

        # Create the classification head
        self._flatten = tf.keras.layers.Flatten(input_shape=input_shape)
        self._classifier = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        # Create the loss, metric and the optimizer
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._accuracy = tf.metrics.SparseCategoricalAccuracy()
        self._optimizer = tf.optimizers.Adam(args.learning_rate)

    def train_epoch(self, dataset):
        for batch in dataset.batches(self._args.batch_size):
            # Forward pass through the convolutions
            hidden = tf.constant(batch["images"])
            convolution_values = [hidden]
            for convolution in self._convolutions:
                hidden = convolution.forward(hidden)
                convolution_values.append(hidden)

            # Run the classification head and compute its gradient
            with tf.GradientTape() as tape:
                tape.watch(hidden)

                predictions = self._flatten(hidden)
                predictions = self._classifier(predictions)
                loss = self._loss(batch["labels"], predictions)

            variables = self._classifier.trainable_variables
            hidden_gradient, *gradients = tape.gradient(loss, [hidden] + variables)

            # Backpropagate the gradient throug the convolutions
            for convolution, inputs, outputs in reversed(
                    list(zip(self._convolutions, convolution_values[:-1], convolution_values[1:]))):
                hidden_gradient, convolution_variables, convolution_gradients = convolution.backward(inputs, outputs,
                                                                                                     hidden_gradient)
                variables.extend(convolution_variables)
                gradients.extend(convolution_gradients)

            # Update the weights
            self._optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, dataset):
        self._accuracy.reset_states()
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"]
            for convolution in self._convolutions:
                hidden = convolution.forward(hidden)
            hidden = self._flatten(hidden)
            predictions = self._classifier(hidden)
            self._accuracy(batch["labels"], predictions)
        return self._accuracy.result().numpy()


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Load data, using only 10000 training images
    mnist = MNIST()
    mnist.train._size = 10000

    # Create the model
    network = Network(args)

    for epoch in range(args.epochs):
        network.train_epoch(mnist.train)

        accuracy = network.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)

    accuracy = network.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)

    # Return the test accuracy for ReCodEx to validate.
    return accuracy


def test():
    c = Convolution(channels=1, kernel_size=2, stride=1, input_shape=(16, 16, 1))
    inputs = tf.initializers.GlorotUniform(seed=42)([1, 16, 16, 1])
    print(c.forward(inputs))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.test:
        test()
        exit(0)

    main(args)
