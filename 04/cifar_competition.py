# Ríša
# 33179fd1-18a0-11eb-8e81-005056ad4f31
# Jaroslava
# a79d74c9-f42b-11e9-9ce9-00505601122b
# Claésia
# 80acc802-7cec-11eb-a1a9-005056ad4f31


# !/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=72, type=int, help="Batch size.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def chunk(layer, kernel, f, layers):
    x = tf.keras.layers.BatchNormalization()(layer)
    x = tf.keras.layers.Conv2D(filters=f, kernel_size=kernel, strides=1, padding='same', activation='relu')(x)

    for i in range(layers):
        conv = tf.keras.layers.Conv2D(filters=f, kernel_size=kernel, strides=1, padding='same', activation='relu')(x)
        add = tf.keras.layers.add([x, conv])
        x = add

    return x


class Network(tf.keras.Model):
    def __init__(self):
        input_layer = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C]) # 32x32x3

        layer_1_bn = tf.keras.layers.BatchNormalization()(input_layer)

        chunk_1 = chunk(layer_1_bn, 2, 64, 7)
        chunk_2 = chunk(chunk_1, 2, 128, 7)
        #chunk_3 = chunk(chunk_2, 2, 256, 7)

        flatten = tf.keras.layers.Flatten()(chunk_2)
        dense = tf.keras.layers.Dense(200, activation='relu')(flatten)
        out = tf.keras.layers.Dense(CIFAR10.LABELS, activation='softmax')(dense)

        super().__init__(inputs=input_layer, outputs=out)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None  # A hack allowing to keep the writers open.

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
    print(f"logdir: {args.logdir}")

    # Load data
    cifar = CIFAR10()

    # TODO: Create the model and train it

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(CIFAR10.W, CIFAR10.H, CIFAR10.C)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(CIFAR10.LABELS, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    tb_callback._close_writers = lambda: None

    model.fit(
        cifar.train.data["images"],
        cifar.train.data["labels"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
        callbacks=[tb_callback],
        shuffle=True
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
