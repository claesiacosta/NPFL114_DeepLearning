#!/usr/bin/env python3

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

from electra_czech_small_lc import ElectraCzechSmallLc
from reading_comprehension_dataset import ReadingComprehensionDataset

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--epochs", default=8, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--epsilon", default=1e-08, type=float)
parser.add_argument("--clipnorm", default=1, type=float)



class Network(tf.keras.Model):
    def __init__(self, electra_model, args):
        inputs = tf.keras.layers.Input([None], dtype=tf.int64, ragged=True)
        attention_mask = tf.sequence_mask(inputs.row_lengths())
        hidden = electra_model(inputs.to_tensor(), attention_mask=attention_mask)[0]

        start = tf.keras.layers.Dense(1, activation=None)(hidden)
        start = tf.keras.layers.Softmax()(tf.squeeze(start, axis=-1))
        end = tf.keras.layers.Dense(1, activation=None)(hidden)
        end = tf.keras.layers.Softmax()(tf.squeeze(end, axis=-1))

        super().__init__(inputs=inputs, outputs={"start":start,"end":end})

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=args.epsilon, clipnorm=args.clipnorm),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics = tf.keras.metrics.SparseCategoricalAccuracy()
        )


def create_dataset(dataset, name, tokenizer):
    inputs = []
    outputs = []
    contexts = []

    for paragraph in dataset.paragraphs:
        context = paragraph["context"]
        context_encoded = tokenizer.encode(context)
        qas = paragraph["qas"]
        for qa in qas:
            qa_encoded = tokenizer.encode(qa["question"])
            inp = context_encoded+qa_encoded[1:]
            if len(inp)>512:
                context_encoded = context_encoded[:512-len(inp)-1] + [context_encoded[-1]]
                inp = context_encoded+qa_encoded[1:]

            if name == "test":
                inputs.append(inp)
                contexts.append(context)

            for answer in qa["answers"]:
                answer_ = answer["text"]
                if len(answer_)==0:
                    continue

                offset = answer["start"]
                token_obj = tokenizer(context)
                start = token_obj.char_to_token(offset)
                end_offset = offset + len(answer_) - 1
                end = token_obj.char_to_token(end_offset)

                if end > len(context_encoded)-2 or start > len(context_encoded)-2:
                    continue

                start_end=(tf.constant(start),tf.constant(end))

                inputs.append(inp)
                outputs.append(start_end)
                contexts.append(context)

    if name == "train" or name == "dev":
        data_tensors = (tf.ragged.constant(inputs), outputs)
        dataset = tf.data.Dataset.from_tensor_slices(data_tensors)
        dataset = dataset.map(lambda x,y: (x, {"start":y[0], "end":y[1]}))
    else:
        data_tensors = tf.ragged.constant(inputs)
        dataset = tf.data.Dataset.from_tensor_slices(data_tensors)

    dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if name == 'test':
        return dataset, contexts
    else:
        return dataset


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

    # Load the Electra Czech small lowercased
    electra = ElectraCzechSmallLc()
    tokenizer = electra.create_tokenizer()
    electra_model = electra.create_model(output_hidden_states=True, trainable = True)

    # Load the data.
    dataset = ReadingComprehensionDataset()

#    ### shorten the sample for debugging purposes like this.
#     dataset.train.paragraphs = dataset.train.paragraphs[0:2]
#     dataset.test.paragraphs = dataset.test.paragraphs[0:2]
#     dataset.dev.paragraphs = dataset.dev.paragraphs[0:2]

    # dataset.train.paragraphs = dataset.train.paragraphs[0:2]
    # dataset.test.paragraphs = dataset.test.paragraphs[0:2]
    # dataset.dev.paragraphs = dataset.dev.paragraphs[0:2]

    train = create_dataset(dataset.train, "train", tokenizer)
    dev = create_dataset(dataset.dev, "dev", tokenizer)
    test, context = create_dataset(dataset.test, "test", tokenizer)

    # TODO: Create the model and train it
    network = Network(electra_model, args)
    network.fit(train, epochs=args.epochs, validation_data=dev)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings (if the answer is not
        # in the context, use an empty string).
        id = 0
        for t in test:
            predictions = network.predict([t])
            starts = predictions["start"]
            ends = predictions["end"]

            for start_id in range(len(starts)):
                start = np.argmax(starts[start_id][1:])+1
                end = min(start + np.argmax(ends[start_id][start:]), len(tokenizer.encode(context[id]))-2)
                encoded = tokenizer(context[id])

                if start <= end:
                    answer = context[id][encoded.token_to_chars(start).start: encoded.token_to_chars(end).end]
                else:
                    answer = ""

                id += 1

                print(answer, file=predictions_file)



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)


