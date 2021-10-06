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

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.1, type=float, help="Mask words with the given probability.")
parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")


class Network(tf.keras.Model):
    # Note that in TF 2.4, computing losses and metrics on RaggedTensors is not yet
    # supported (it will be in TF 2.5). Therefore, we override the `train_step` method
    # to support it, passing the "flattened" predictions and gold data to the loss
    # and metrics.
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Check that both the gold data and predictions are RaggedTensors.
            assert isinstance(y_pred, tf.RaggedTensor) and isinstance(y, tf.RaggedTensor)
            loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}

    # Analogously to `train_step`, we also need to override `test_step`.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}
        


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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # TODO: Create the model and train it
    words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)
    words_id = morpho.train.forms.word_mapping(words)
    num_words = morpho.train.forms.word_mapping.vocab_size()
    num_chars = morpho.train.forms.char_mapping.vocab_size()
    num_tags = morpho.train.tags.word_mapping.vocab_size()

    ones_like = tf.ones_like(words_id, dtype=tf.float32)
    dropout = tf.keras.layers.Dropout(args.word_masking)(ones_like)
    words_id = words_id * tf.cast(dropout != 0, tf.int64)

    embed_words = tf.keras.layers.Embedding(num_words, args.we_dim)(words_id)

    words_uni, indices_uni = tf.unique(words.values)

    chars_seq = tf.strings.unicode_split(words_uni, input_encoding='UTF-8')

    chars_id = morpho.train.forms.char_mapping(chars_seq)

    embed_chars = tf.keras.layers.Embedding(num_chars, args.cle_dim)(chars_id)

    hidden_char = tf.keras.layers.GRU(args.cle_dim)
    hidden_char = tf.keras.layers.Bidirectional(hidden_char)(embed_chars)#(embed_chars.to_tensor(), mask=tf.sequence_mask(embed_chars.row_lengths()))
    #hidden_char = tf.RaggedTensor.from_tensor(hidden_char, embed_chars.row_lengths())


    flattened = tf.gather(hidden_char, indices_uni)

    ragged = words.with_values(flattened)

    concat_embed = tf.concat([embed_words, ragged], -1)


    hidden = tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True)
    hidden = tf.keras.layers.Bidirectional(hidden, merge_mode="sum")(concat_embed.to_tensor(), mask=tf.sequence_mask(concat_embed.row_lengths()))
    hidden = tf.RaggedTensor.from_tensor(hidden, concat_embed.row_lengths())

    predictions = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_tags, activation=tf.nn.softmax))(hidden)

    assert predictions.shape.rank == 3

    def tagging_dataset(forms, lemmas, tags):
        return forms, morpho.train.tags.word_mapping(tags)

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(tagging_dataset)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

#    model = tf.keras.Model(inputs=words, outputs=predictions)
    model = Network(inputs=words, outputs=predictions)
    print(model.summary())
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=0, update_freq=100, profile_batch=0)

    model.fit(
        train,
        epochs=args.epochs,
        validation_data=dev,
        callbacks=[tb_callback]
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set; update the following prediction
        # command if you use other output structre than in tagger_we.
        predictions = model.predict(test)
        tag_strings = morpho.test.tags.word_mapping.get_vocabulary()
        for sentence in predictions:
            for word in sentence:
                print(tag_strings[np.argmax(word)], file=predictions_file)
            print(file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
