"""Reference model. Run after installing TensorFlow: python train.py"""


def build_model(frames: int = 30, features: int = 63, classes: int = 12):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(frames, features), name="landmark_sequence")
    x = tf.keras.layers.Masking()(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.GRU(48)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax", name="sign")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_model().summary()
