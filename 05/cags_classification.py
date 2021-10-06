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

from cags_dataset import CAGS
import efficient_net

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def train_aug(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 6, CAGS.W + 6)
    image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 12, dtype=tf.int32),
                                        tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 12, dtype=tf.int32)])
    #image = (image / 255.0)
    image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])
    return image, label

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

    # Load the data
    cags = CAGS()

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    efficientnet_b0.trainable = False

    # TODO: Prepare data

    train = cags.train.map(lambda example: (example["image"], example["label"]))
    train = train.map(train_aug)
    train = train.shuffle(10000, seed=args.seed).batch(args.batch_size)

    dev = cags.dev.map(lambda example: (example["image"], example["label"])).batch(args.batch_size)
    test = cags.test.map(lambda example: example["image"]).batch(args.batch_size)

    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="image")
    x = efficientnet_b0(inputs, training=False)
    x = tf.keras.layers.Flatten()(x[0])
    x = tf.keras.layers.Dense(200, activation='relu')(x)
    outputs = tf.keras.layers.Dense(len(CAGS.LABELS), activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    model.fit(
        train,
        epochs=args.epochs,
        validation_data=dev,
        callbacks=[tb_callback]
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
