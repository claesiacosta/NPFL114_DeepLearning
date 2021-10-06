import tensorflow as tf

inputs = tf.keras.Input(shape=(1,), name="input_1")
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
label_1 = tf.keras.layers.Dense(7, activation=tf.nn.softmax, name="label")(x)
label_2 = tf.keras.layers.Dense(7, activation=tf.nn.softmax, name="other_label")(x)

outputs = {
    "label": label_1,
    "other_label": label_2
}

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.output_names = sorted(outputs.keys())

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss={
                  "label": tf.keras.losses.SparseCategoricalCrossentropy(),
                  "other_label": tf.keras.losses.SparseCategoricalCrossentropy()
              },
              metrics={
                  "label": [tf.metrics.BinaryAccuracy(name="accuracy")],
                  "other_label": [tf.metrics.BinaryAccuracy(name="accuracy")]
              })

dt = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3, 4, 5, 6])
dt = dt.map(lambda i: float(i))
dt = dt.map(lambda i: (i, {"label": i, "other_label": i}))
dt = dt.batch(1)

model.fit(dt)
