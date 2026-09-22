"""Fixed CNN architecture and optimizer construction."""

from __future__ import annotations

from typing import Any


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for CNN evaluation. Install it with "
            "`pip install tensorflow`."
        ) from error
    return tf


def build_model(configuration: dict[str, Any], mode: str):
    tf = _tensorflow()
    layers = tf.keras.layers

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(configuration["dropout_1"]),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.BatchNormalization(),
            layers.Dense(configuration["dense_units_1"], activation=configuration["activation"]),
            layers.Dropout(configuration["dropout_2"]),
            layers.Dense(configuration["dense_units_2"], activation=configuration["activation"]),
            layers.Dense(10, activation="softmax"),
        ]
    )

    optimizer_name = configuration["optimizer"]
    optimizer_classes = {
        "adam": tf.keras.optimizers.Adam,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "sgd": tf.keras.optimizers.SGD,
        "adadelta": tf.keras.optimizers.Adadelta,
        "adagrad": tf.keras.optimizers.Adagrad,
        "adamax": tf.keras.optimizers.Adamax,
        "ftrl": tf.keras.optimizers.Ftrl,
        "nadam": tf.keras.optimizers.Nadam,
    }
    optimizer_kwargs: dict[str, Any] = {}
    if mode == "improved":
        optimizer_kwargs["learning_rate"] = configuration["learning_rate"]
    try:
        optimizer_class = optimizer_classes[optimizer_name]
    except KeyError as error:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}") from error
    optimizer = optimizer_class(**optimizer_kwargs)

    loss = configuration.get("loss", "sparse_categorical_crossentropy")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["sparse_categorical_accuracy"],
    )
    return model, float(optimizer.learning_rate.numpy())
