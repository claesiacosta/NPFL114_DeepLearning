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
import tensorflow_addons as tfa

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=800, type=int, help="CLE embedding dimension.")
parser.add_argument("--rnn_dim", default=800, type=int, help="RNN cell dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class Network(tf.keras.Model):
    def __init__(self, args, train):
        super().__init__()

        self.source_mapping = train.forms.char_mapping
        self.target_mapping = train.lemmas.char_mapping
        self.target_mapping_inverse = type(self.target_mapping)(
            mask_token=None, vocabulary=self.target_mapping.get_vocabulary(), invert=True)

        # TODO(lemmatizer_noattn): Define
        # - `self.source_embedding` as an embedding layer of source chars into `args.cle_dim` dimensions
        self.source_embeddings = tf.keras.layers.Embedding(self.source_mapping.vocab_size(), args.cle_dim)
        # TODO: Define
        # - `self.source_rnn` as a bidirectional GRU with `args.rnn_dim` units, returning **whole sequences**,
        #   summing opposite directions
        self.source_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.rnn_dim, return_sequences=True), merge_mode='sum')
        # TODO(lemmatizer_noattn): Then define
        # - `self.target_embedding` as an embedding layer of target chars into `args.cle_dim` dimensions
        # - `self.target_rnn_cell` as a GRUCell with `args.rnn_dim` units
        # - `self.target_output_layer` as a Dense layer into as many outputs as there are unique target chars
        self.target_embedding = tf.keras.layers.Embedding(self.target_mapping.vocab_size(), args.cle_dim)
        self.target_rnn_cell = tf.keras.layers.GRUCell(args.rnn_dim)
        self.target_output_layer = tf.keras.layers.Dense(self.target_mapping.vocab_size(), activation=None)
        # TODO: Define
        # - `self.attention_source_layer` as a Dense layer with `args.rnn_dim` outputs
        # - `self.attention_state_layer` as a Dense layer with `args.rnn_dim` outputs
        # - `self.attention_weight_layer` as a Dense layer with 1 output
        self.attention_source_layer = tf.keras.layers.Dense(args.rnn_dim)
        self.attention_state_layer = tf.keras.layers.Dense(args.rnn_dim)
        self.attention_weight_layer = tf.keras.layers.Dense(1)
        # Compile the model
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.Accuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.

    class DecoderTraining(tfa.seq2seq.BaseDecoder):
        def __init__(self, lemmatizer, *args, **kwargs):
            self.lemmatizer = lemmatizer
            super().__init__.__wrapped__(self, *args, **kwargs)

        @property
        def batch_size(self):
            # TODO(lemmatizer_noattn): Return the batch size of self.source_states, using tf.shape
            return tf.shape(self.source_states)[0]
        @property
        def output_size(self):
            # TODO(lemmatizer_noattn): Return `tf.TensorShape(number of logits per each output element)`
            # By output element we mean characters.
            with tf.init_scope():
                return tf.TensorShape(self.lemmatizer.target_mapping.vocab_size())
        @property
        def output_dtype(self):
            # TODO(lemmatizer_noattn): Return the type of the logits
            return tf.float32

        def with_attention(self, inputs, states):
            # TODO: Compute the attention.
            # - Take self.source_states` and pass it through the self.lemmatizer.attention_source_layer.
            #   Because self.source_states does not change, you should in fact do it in `initialize`.
            # - Pass `states` though `self.lemmatizer.attention_state_layer`.
            # - Sum the two outputs. However, the first has shape [a, b, c] and the second [a, c]. Therefore,
            #   expand the second to [a, b, c] or [a, 1, c] (the latter works because of broadcasting rules).
            # - Pass the sum through `tf.tanh` and through the `self.lemmatizer.attention_weight_layer`.
            # - Then, run softmax on a suitable axis, generating `weights`.
            # - Multiply `self.source_states` with `weights` and sum the result in the axis
            #   corresponding to characters, generating `attention`. Therefore, `attention` is a a fixed-size
            #   representation for every batch element, independently on how many characters had
            #   the corresponding input forms.
            # - Finally concatenate `inputs` and `attention` (in this order) and return the result.
            source_states = self.lemmatizer.attention_source_layer(self.source_states)
            states_pass = self.lemmatizer.attention_state_layer(states)
            weights = tf.nn.softmax(self.lemmatizer.attention_weight_layer(tf.tanh(source_states + tf.expand_dims(states_pass, axis=1))), axis=1)
            attention = tf.reduce_sum(tf.math.multiply(self.source_states, weights), axis=1)
            return tf.concat([inputs, attention], axis=1)

        def initialize(self, layer_inputs, initial_state=None, mask=None):
            self.source_states, self.targets = layer_inputs
            # TODO(lemmatizer_noattn): Define `finished` as a vector of self.batch_size of `False` [see tf.fill].
            finished = tf.fill([self.batch_size], False)
            # TODO(lemmatizer_noattn): Define `inputs` as a vector of self.batch_size of MorphoDataset.Factor.BOW,
            # embedded using self.lemmatizer.target_embedding
            inputs = self.lemmatizer.target_embedding(tf.fill([self.batch_size], MorphoDataset.Factor.BOW))
            # TODO: Define `states` as the representation of the first character
            # in `source_states`. The idea is that it is most relevant for generating
            # the first letter and contains all following characters via the backward RNN.
            states = self.source_states[:, 0, :]
            # TODO: Pass `inputs` through `self.with_attention(inputs, states)`.
            inputs = self.with_attention(inputs, states)
            return finished, inputs, states

        def step(self, time, inputs, states, training):
            # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.
            (outputs, [states]) = self.lemmatizer.target_rnn_cell(inputs, [states])
            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
            outputs = self.lemmatizer.target_output_layer(outputs)
            # TODO(lemmatizer_noattn): Define `next_inputs` by embedding `time`-th chars from `self.targets`.
            next_inputs = self.lemmatizer.target_embedding(self.targets[:, time])
            # TODO(lemmatizer_noattn): Define `finished` as True if `time`-th char from `self.targets` is
            # `MorphoDataset.Factor.EOW`, False otherwise.
            finished = tf.equal(self.targets[:,time], MorphoDataset.Factor.EOW)
            # TODO: Pass `next_inputs` through `self.with_attention(next_inputs, states)`.
            next_inputs = self.with_attention(next_inputs, states)
            return outputs, states, next_inputs, finished

    class DecoderPrediction(DecoderTraining):
        @property
        def output_size(self):
            # TODO(lemmatizer_noattn): Return `tf.TensorShape()` describing a scalar element,
            # because we are generating scalar predictions now.
            return tf.TensorShape([])
        @property
        def output_dtype(self):
            # TODO(lemmatizer_noattn): Return the type of the generated predictions
            return tf.int32

        def initialize(self, layer_inputs, initial_state=None, mask=None):
            # Use `initialize` from the DecoderTraining, passing None as targets
            return super().initialize([layer_inputs, None], initial_state)

        def step(self, time, inputs, states, training):
            # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.
            (outputs, [states]) = self.lemmatizer.target_rnn_cell(inputs, [states])
            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
            outputs = self.lemmatizer.target_output_layer(outputs)
            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through `tf.argmax` on suitable axis and with
            # `output_type=tf.int32` parameter.
            outputs = tf.argmax(outputs, axis=1, output_type=tf.int32)
            # TODO(lemmatizer_noattn): Define `next_inputs` by embedding the `outputs`
            next_inputs = self.lemmatizer.target_embedding(outputs)
            # TODO(lemmatizer_noattn): Define `finished` as True if `outputs are `MorphoDataset.Factor.EOW`, False otherwise.
            finished = tf.equal(outputs, MorphoDataset.Factor.EOW)
            # TODO(DecoderTraining): Pass `next_inputs` through `self.with_attention(next_inputs, states)`.
            next_inputs = self.with_attention(next_inputs, states)
            return outputs, states, next_inputs, finished

    # If `targets` is given, we are in the teacher forcing mode.
    # Otherwise, we run in autoregressive mode.
    def call(self, inputs, targets=None):
        # FIX: Get indices of valid lemmas and reshape the `source_charseqs`
        # so that it is a list of valid sequences, instead of a
        # matrix of sequences, some of them padding ones.
        source_charseqs = inputs.values
        source_charseqs = tf.strings.unicode_split(source_charseqs, "UTF-8")
        source_charseqs = self.source_mapping(source_charseqs)
        if targets is not None:
            target_charseqs = targets.values
            target_charseqs = target_charseqs.to_tensor()

        # TODO(lemmatizer_noattn): Embed source_charseqs using `source_embedding`
        source_embedded = self.source_embeddings(source_charseqs)
        # TODO: Run source_rnn on the embedded sequences, returning outputs in `source_states`.
        # However, convert the embedded sequences from a RaggedTensor to a dense Tensor first,
        # i.e., call the source_rnn with
        # `(source_embedded.to_tensor(), mask=tf.sequence_mask(source_embedded.row_lengths()))`.
        source_states = self.source_rnn(source_embedded.to_tensor(), mask=tf.sequence_mask(source_embedded.row_lengths()))

        # Run the appropriate decoder
        if targets is not None:
            # TODO(lemmatizer_noattn): Create a self.DecoderTraining by passing `self` to its constructor.
            # Then run it on `[source_states, target_charseqs]` input,
            # storing the first result in `output` and the third result in `output_lens`.
            #print("source_states ", source_states.shape)
            output, aaa, output_lens = self.DecoderTraining(self)([source_states, target_charseqs])
        else:
            # TODO(lemmatizer_noattn): Create a self.DecoderPrediction by using:
            # - `self` as first argument to its constructor
            # - `maximum_iterations=tf.cast(source_charseqs.bounding_shape(1) + 10, tf.int32)`
            #   as another argument, which indicates that the longest prediction
            #   must be at most 10 characters longer than the longest input.
            # Then run it on `source_states`, storing the first result
            # in `output` and the third result in `output_lens`.
            # Finally, because we do not want to return the `[EOW]` symbols,
            # decrease `output_lens` by one.
            output, aaa, output_lens =  self.DecoderPrediction(self, maximum_iterations=tf.cast(source_charseqs.bounding_shape(1) + 10, tf.int32))(source_states)
            output_lens = output_lens-1

        # Reshape the output to the original matrix of lemmas
        # and explicitly set mask for loss and metric computation.
        output = tf.RaggedTensor.from_tensor(output, output_lens)
        output = inputs.with_values(output)
        return output

    def train_step(self, data):
        x, y = data

        # Convert `y` by splitting characters, mapping characters to ids using
        # `self.target_mapping` and finally appending `MorphoDataset.Factor.EOW`
        # to every sequence.
        y_targets = self.target_mapping(tf.strings.unicode_split(y.values, "UTF-8"))
        y_targets = tf.concat(
            [y_targets, tf.fill([y_targets.bounding_shape(0), 1], tf.constant(MorphoDataset.Factor.EOW, tf.int64))], axis=-1)
        y_targets = y.with_values(y_targets)

        with tf.GradientTape() as tape:
            y_pred = self(x, targets=y_targets, training=True)
            loss = self.compiled_loss(y_targets.flat_values, y_pred.flat_values, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    def predict_step(self, data):
        if isinstance(data, tuple): data = data[0]
        y_pred = self(data, training=False)
        y_pred = self.target_mapping_inverse(y_pred)
        y_pred = tf.strings.reduce_join(y_pred, axis=-1)
        return y_pred

    def test_step(self, data):
        x, y = data
        y_pred = self.predict_step(data)
        self.compiled_metrics.update_state(tf.ones_like(y.values, dtype=tf.int32), tf.cast(y_pred.values == y.values, tf.int32))
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
    morpho = MorphoDataset("czech_pdt", add_bow_eow=True)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # TODO: Create the model and train it
        # Create the network and train
    model = Network(args, morpho.train)

    # Construct dataset for lemmatizer training
    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(lambda forms, lemmas, tags: (forms, lemmas))
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    # Callback showing intermediate results during training
    class ShowIntermediateResults(tf.keras.callbacks.Callback):
        def __init__(self, data):
            self._iterator = iter(data.repeat())
        def on_train_batch_end(self, batch, logs=None):
            if model.optimizer.iterations % 10 == 0:
                forms, lemmas = next(self._iterator)
                tf.print(model.optimizer.iterations, forms[0, 0], lemmas[0, 0], model.predict_on_batch(forms[:1, :1])[0, 0])

    model.fit(train, epochs=args.epochs, validation_data=dev, verbose=2,
                callbacks=[ShowIntermediateResults(dev), model.tb_callback])

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "lemmatizer_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set; update the following prediction
        # command if you use other output structre than in lemmatizer_noattn.
        predictions = model.predict(test)
        for sentence in predictions:
            for word in sentence:
                print(word.numpy().decode("utf-8"), file=predictions_file)
            print(file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
