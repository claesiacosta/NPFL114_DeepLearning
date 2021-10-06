#!/usr/bin/env python3
###
### Team members:
## Jaroslava Schovancova
# a79d74c9-f42b-11e9-9ce9-00505601122b
## Antonia Claésia da Costa Souza
# 80acc802-7cec-11eb-a1a9-005056ad4f31
## Richard Hajek
# 33179fd1-18a0-11eb-8e81-005056ad4f31
###
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
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--model", default="model_segment_3.h5", type=str, help="Output model path.")
parser.add_argument("--finetuned_model", default="model_segment_3_finetuned.h5")
parser.add_argument("--finetuned_epochs", default=60, type=int)


### Materials
### * https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
### * https://data-flair.training/blogs/cats-dogs-classification-deep-learning-project-beginners/


### Data variation for training
def train_variation(image, mask):
    """
    Variations of the image:
    * horizontal flip
    * adjust "colors": hue, saturation, brightness, contrast
    """
    PROBABILITY_HORIZONTAL_FLIP = 0.5
    RANDOM_ADJUST_HUE = 0.08
    RANDOM_ADJUST_BRIGHTNESS = 0.05
    RANDOM_ADJUST_SATURATION_RANGE = 0.4
    RANDOM_ADJUST_CONTRAST_RANGE = 0.3

    ### horizontal flip
    if tf.random.uniform(()) > PROBABILITY_HORIZONTAL_FLIP:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    ### Adjust hue
    image = tf.image.random_hue(image, RANDOM_ADJUST_HUE)

    ### Adjust saturation
    image = tf.image.random_saturation(image, 1.0-RANDOM_ADJUST_SATURATION_RANGE, 1.0+RANDOM_ADJUST_SATURATION_RANGE)

    ### Adjust brightness
    image = tf.image.random_brightness(image, RANDOM_ADJUST_BRIGHTNESS)

    ### Adjust contrast
    image = tf.image.random_contrast(image, 1.0-RANDOM_ADJUST_CONTRAST_RANGE, 1.0+RANDOM_ADJUST_CONTRAST_RANGE)

    return image, mask


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

    ### Load data, then decode it, will become dict with keys "image", "mask", "label"; we don't use "label" so forget it
    dataset = {}
    for k in ["train", "dev", "test"]:
        dataset[k] = tf.data.TFRecordDataset(f"cags.{k}.tfrecord").map(CAGS.parse).map(lambda example: (example["image"], example["mask"]))

    ### Data pipelines
    pipeline = {}
    pipeline["train"] = dataset["train"].shuffle(5000, seed=args.seed).map(train_variation)
    for k in ["train", "dev", "test"]:
        pipeline[k] = dataset[k].batch(args.batch_size)

    ### Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    efficientnet_b0.trainable = False

    # TODO: Create the model and train it
    inputs = tf.keras.layers.Input([CAGS.H, CAGS.W, CAGS.C])
    features = efficientnet_b0(inputs)

    x = features[1]
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for feature in features[2:]:
        x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(rate=0.25)(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(rate=0.25)(x)
        f = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(feature)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.ReLU()(f)
        x = tf.keras.layers.Dropout(rate=0.25)(x)
        x = tf.keras.layers.Add()([x, f])

    outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), CAGS.MaskIoUMetric(name="iou") ],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    #### Print model summary
    model.summary()

    model.fit(
        x=pipeline["train"],
        epochs=args.epochs,
        validation_data=pipeline["dev"],
        callbacks=[tb_callback]
    )

    ### Save the model
    model.save(args.model)

    ### fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        CAGS.MaskIoUMetric(name="iou")
        ],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    ### Print model summary
    model.summary()

    ### Train the model
    model.fit(
        x=pipeline["train"],
        epochs=args.finetuned_epochs,
        initial_epoch=args.epochs,
        validation_data=pipeline["dev"],
        callbacks=[tb_callback]
    )

    ### Save the model
    model.save(args.finetuned_model)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
#         test_masks = model.predict(...)
        test_masks = model.predict(pipeline["test"])

        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)


