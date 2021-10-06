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
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--discriminator_layers", default=[128], type=int, nargs="+", help="Discriminator layers.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--generator_layers", default=[128], type=int, nargs="+", help="Generator layers.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
# If you add more arguments, ReCodEx will keep them with your default values.

# The GAN model
class GAN(tf.keras.Model):
    def __init__(self, args):
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = tfp.distributions.Normal(tf.zeros(args.z_dim), tf.ones(args.z_dim))

        # TODO: Define `self.generator` as a Model, which
        # - takes vectors of [args.z_dim] shape on input
        # - applies len(args.generator_layers) dense layers with ReLU activation,
        #   i-th layer with args.generator_layers[i] units
        # - applies output dense layer with MNIST.H * MNIST.W * MNIST.C units
        #   and sigmoid activation
        # - reshapes the output (tf.keras.layers.Reshape) to [MNIST.H, MNIST.W, MNIST.C]
        vec_input = tf.keras.layers.Input([args.z_dim])
        hidden = vec_input
        for l_size in args.generator_layers:
            hidden = tf.keras.layers.Dense(l_size, activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Dense(MNIST.H * MNIST.W * MNIST.C, activation=tf.nn.sigmoid)(hidden)
        output = tf.keras.layers.Reshape([MNIST.H, MNIST.W, MNIST.C])(hidden)
        self.generator = tf.keras.Model(inputs=vec_input, outputs=output)

        # TODO: Define `self.discriminator` as a Model, which
        # - takes input images with shape [MNIST.H, MNIST.W, MNIST.C]
        # - flattens them
        # - applies len(args.discriminator_layers) dense layers with ReLU activation,
        #   i-th layer with args.discriminator_layers[i] units
        # - applies output dense layer with one output and a suitable activation function
        img_input = tf.keras.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        hidden = tf.keras.layers.Flatten()(img_input)
        for l_size in args.discriminator_layers:
            hidden = tf.keras.layers.Dense(l_size, activation=tf.nn.relu)(hidden)
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
        self.discriminator = tf.keras.Model(inputs=img_input, outputs=output)

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq="epoch", profile_batch=0)

    # We override `compile`, because we want to use two optimizers.
    def compile(self, discriminator_optimizer, generator_optimizer):
        super().compile(
            loss=tf.losses.BinaryCrossentropy(),
            metrics=tf.metrics.BinaryAccuracy("discriminator_accuracy"),
        )

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

    def train_step(self, images):
        with tf.GradientTape() as tape:
        # TODO: Generator training. With a Gradient tape:
        # - generate as many random latent samples as there are images by a single call
        #   to `self._z_prior.sample`; also pass `seed=self._seed` for replicability;
        # - pass the samples through a generator; do not forget about `training=True`
        # - run discriminator on the generated images, also using `training=True` (even if
        #   not updating discriminator parameters, we want to perform possible BatchNorm in it)
        # - compute `generator_loss` using `self.compiled_loss`, with ones as target labels
        #   (`tf.ones_like` might come handy).
        # Then, run an optimizer step with respect to generator trainable variables.
            z = self._z_prior.sample(sample_shape=tf.shape(images)[0], seed=self._seed)
            generator_img = self.generator(z, training=True)
            discriminator_out = self.discriminator(generator_img,  training=True)
            generator_loss = self.compiled_loss(tf.ones_like(discriminator_out), discriminator_out)

        variables = self.generator.trainable_variables
        grad = tape.gradient(generator_loss, variables)
        self.generator_optimizer.apply_gradients(zip(grad, variables))

        with tf.GradientTape() as tape:
        # TODO: Discriminator training. Using a Gradient tape:
        # - discriminate `images` with `training=True`, storing
        #   results in `discriminated_real`
        # - discriminate images generated in generator training with `training=True`,
        #   storing results in `discriminated_fake`
        # - compute `discriminator_loss` by summing
        #   - `self.compiled_loss` on `discriminated_real` with suitable targets,
        #   - `self.compiled_loss` on `discriminated_fake` with suitable targets.
        # Then, run an optimizer step with respect to discriminator trainable variables.
            discriminated_real = self.discriminator(images, training=True)
            discriminated_fake = self.discriminator(generator_img, training=True)
            loss_real = self.compiled_loss(tf.ones_like(discriminated_real), discriminated_real)
            loss_fake = self.compiled_loss(tf.zeros_like(discriminated_fake), discriminated_fake)
            discriminator_loss = loss_real + loss_fake
            #discriminator_loss = tf.reduce_sum(loss_real) + tf.reduce_sum(loss_fake)
        variables = self.discriminator.trainable_variables
        grad = tape.gradient(discriminator_loss, variables)
        self.discriminator_optimizer.apply_gradients(zip(grad, variables))

        # TODO: Update the discriminator accuracy metric -- call the
        # `self.compiled_metrics.update_state` twice, with the same arguments
        # the `self.compiled_loss` was called during discriminator loss computation.
        self.compiled_metrics.update_state(tf.ones_like(discriminated_real), discriminated_real)
        self.compiled_metrics.update_state(tf.zeros_like(discriminated_fake), discriminated_fake)

        return {
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
            "loss": discriminator_loss + generator_loss,
            **{metric.name: metric.result() for metric in self.metrics}
        }

    def generate(self, epoch, logs):
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.generator(self._z_prior.sample(GRID * GRID, seed=self._seed), training=False)

        # Generate GRIDxGRID interpolated images
        if self._z_dim == 2:
            # Use 2D grid for sampled Z
            starts = tf.stack([-2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
            ends = tf.stack([2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
        else:
            # Generate random Z
            starts, ends = self._z_prior.sample(GRID, seed=self._seed), self._z_prior.sample(GRID, seed=self._seed)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.expand_dims(tf.linspace(0., 1., GRID), -1) for i in range(GRID)], axis=0)
        interpolated_images = self.generator(interpolated_z, training=False)

        # Stack the random images, then an empty row, and finally interpolated imates
        image = tf.concat(
            [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
            [tf.zeros([MNIST.H, MNIST.W * GRID, MNIST.C])] +
            [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0)
        with self.tb_callback._train_writer.as_default(step=epoch):
            tf.summary.image("images", tf.expand_dims(image, 0))

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

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST(args.dataset)

    # Create the network and train
    network = GAN(args)
    network.compile(
        discriminator_optimizer=tf.optimizers.Adam(),
        generator_optimizer=tf.optimizers.Adam(),
    )
    logs = network.fit(
        mnist.train.dataset.map(lambda example: example["images"]).shuffle(mnist.train.size, args.seed).batch(args.batch_size),
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_epoch_end=network.generate),
            network.tb_callback,
        ]
    )

    # Return loss and discriminator accuracy for ReCodEx to validate
    return logs.history["loss"][-1], logs.history["discriminator_accuracy"][-1]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
