### Team members:
## Jaroslava Schovancova
# a79d74c9-f42b-11e9-9ce9-00505601122b
## Antonia Claésia da Costa Souza
# 80acc802-7cec-11eb-a1a9-005056ad4f31
## Richard Hajek
# 33179fd1-18a0-11eb-8e81-005056ad4f31
###


#!/usr/bin/env python3w
import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import bboxes_utils
import efficient_net
from svhn_dataset import SVHN

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
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
    svhn = SVHN()
    SVHN_W, SVHN_H, CHANNELS = 224, 224, 3
    anchors = []
    corner1 = SVHN_W // 16
    corner2 = SVHN_W // corner1
    for i in range(corner1):
        for j in range(corner1):
            anchors.append(tf.stack([
                i * corner2 - (SVHN_H // 2) // 2,
                j * corner2 - (SVHN_W // 2) // 2,
                i * corner2 + (SVHN_H // 2) // 2,
                j * corner2 + (SVHN_W // 2) // 2
            ]))

    anchors = tf.stack(anchors, axis=0)

    def create_dataset(name):
        def prepare_example(example):

            # Get image info
            image_height = tf.cast(tf.shape(example["image"])[0], tf.float32)
            image_width = tf.cast(tf.shape(example["image"])[1], tf.float32)

            # Adjust the bboxes coordinates to scale 0 to 1
            ratio = tf.stack((image_height, image_width, image_height, image_width), axis=0)
            bboxes = example["bboxes"] / ratio

            # Resize image to 224x224, resize the bboxes to match
            img = tf.image.resize(example["image"], [224, 224])
            bboxes = bboxes * 224
            bboxes = tf.floor(bboxes)

            anchors_classes, anchors_bboxes = tf.numpy_function(bboxes_utils.bboxes_training,
                                                                [anchors, example["classes"], bboxes, 0.7],
                                                                [tf.double, tf.double])

            anchors_classes = tf.cast(anchors_classes, tf.int32)
            anchors_bboxes = tf.cast(anchors_bboxes, tf.int32)

            ex_weight_classes = 1
            ex_weight_bboxes = tf.where(0 < anchors_classes, 1, 0)
            anchors_classes = tf.one_hot(anchors_classes - 1, 10)
            return img, (anchors_classes, anchors_bboxes), (ex_weight_classes, ex_weight_bboxes)

        # setattr(svhn, name, getattr(svhn, name).map(svhn.parse))
        dataset = getattr(svhn, name)

        dataset = dataset.map(prepare_example).cache()

        if name == "train":
            dataset = dataset.shuffle(len(dataset), seed=args.seed)

        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(args.threads)
        return dataset

    train = create_dataset("train")
    dev = create_dataset("dev")
    test = create_dataset("test")
    test_org_sizes = svhn.test.map(lambda x: tf.shape(x["image"]))

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False, dynamic_input_shape=True)
    efficientnet_b0.trainable = False

    # TODO: Create the model and train it

    inputs = tf.keras.layers.Input([SVHN_H, SVHN_W, CHANNELS])
    features = efficientnet_b0(inputs)

    head = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(features[2])
    head = tf.keras.layers.BatchNormalization()(head)
    head = tf.keras.layers.ReLU()(head)

    head = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(head)
    head = tf.keras.layers.BatchNormalization()(head)
    head = tf.keras.layers.ReLU()(head)

    head = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=1, padding='same', activation=tf.nn.sigmoid)(head)
    head = tf.reshape(head, [-1, corner1 * corner1, 10], name="classification_head")

    reg = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(features[2])
    reg = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(reg)
    reg = tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=1, padding='same')(reg)
    #reg = tf.keras.layers.BatchNormalization()(reg)

    reg = tf.reshape(reg, [-1, corner1 * corner1, 4], name="regression_head")

    model = tf.keras.Model(inputs=inputs, outputs=(head, reg))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=(
            tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
              tf.keras.losses.Huber()),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    model.summary()

    model.fit(
        train,
        epochs=args.epochs,
        validation_data=dev,
        callbacks=[tb_callback]
    )

    predict = model.predict(test)

    pred_class_box = []
    for i, org_size in zip(range(predict[1].shape[0]), test_org_sizes):
        bboxes = bboxes_utils.bboxes_from_fast_rcnn(anchors, predict[1][i, :, :])

        ratio = tf.stack((SVHN_H, SVHN_W, SVHN_H, SVHN_W), axis=0)
        print(f"Stack ratio: {ratio}")
        ratio = tf.cast(ratio, tf.float32)
        print(f"cast stack ratio: {ratio}")

        res_sco = np.max(predict[0][i, :, :], axis=-1)
        arg_max = np.argmax(predict[0][i, :, :], axis=-1)

        x = tf.image.non_max_suppression(bboxes, res_sco, 5, 0.3)
        ex_bboxes = []
        ex_classes = []

        for j in x:
            if predict[0][i, j, arg_max[j]] > 0.2:
                ex_classes.append(arg_max[j])
                ex_bboxes.append(bboxes[j])

        def convert(bbox):
            bbox_tf = tf.cast(tf.constant(bbox), tf.float32)
            print(f"bboxes_tf: {bbox_tf}")
            bbox_tf = bbox_tf / ratio
            print(f"bboxes_tf after ratio div: {bbox_tf}")
            original_size = tf.cast(tf.stack((org_size[0], org_size[1], org_size[0], org_size[1])), tf.float32)
            print(f"original size: {original_size}")
            bbox_tf *= original_size
            print(f"bboxes_tf final: {bbox_tf}")
            return bbox_tf.numpy().tolist()

        ex_bboxes = [convert(bbox) for bbox in ex_bboxes]

        pred_class_box.append((ex_classes, ex_bboxes))
        print("Predicted classes:", ex_classes, "predicted bboxes:", ex_bboxes)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        for predicted_classes, predicted_bboxes in pred_class_box:
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [label] + list(bbox)
            print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
