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

from modelnet import ModelNet

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

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
    modelnet = ModelNet(args.modelnet)

    #https://www.researchgate.net/publication/332016103_Deep_Learning_for_Validating_and_Estimating_Resolution_of_Cryo-Electron_Microscopy_Density_Maps
    # TODO: Create the model and train it
    inputs = tf.keras.layers.Input(shape=(modelnet.H, modelnet.W, modelnet.D, modelnet.C))

    hidden = tf.keras.layers.Conv3D(32, (7,7,7), activation=None, padding='same')(inputs)
    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

    hidden = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=2)(hidden)

    hidden = tf.keras.layers.Conv3D(64, (5,5,5), activation=None, padding='same')(hidden)
    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

    hidden = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=2)(hidden)

    hidden = tf.keras.layers.Conv3D(128, (5,5,5), activation=None, padding='same')(hidden)
    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

    hidden = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=2)(hidden)

    hidden = tf.keras.layers.Dense(1024, activation='tanh')(hidden)
    hidden = tf.keras.layers.Dropout(rate=0.4)(hidden)
    hidden = tf.keras.layers.GlobalAveragePooling3D()(hidden)
    output = tf.keras.layers.Dense(len(modelnet.LABELS), activation=tf.nn.softmax)(hidden)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
     )
    model.summary()
    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
    tb_callback.on_train_end = lambda *_: None

    model.fit(
        modelnet.train.data['voxels'],
        modelnet.train.data['labels'],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(modelnet.dev.data['voxels'], modelnet.dev.data['labels'])
    )


    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(modelnet.test.data["voxels"])

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
