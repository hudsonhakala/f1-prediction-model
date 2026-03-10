import numpy as np
import tensorflow as tf


def make_data(num_samples: int = 1000):
    # We create a toy dataset with 2 input features per sample.
    # Think of each row as a point: [x1, x2].
    x = np.random.randn(num_samples, 2).astype(np.float32)

    # Create labels (the "correct answer" for each point).
    # If x1 + x2 > 0, label is 1.0; otherwise label is 0.0.
    # This gives us a simple line-based decision boundary.
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)

    # Return:
    # x shape -> (num_samples, 2)
    # y shape -> (num_samples,)
    return x, y


def build_model() -> tf.keras.Model:
    # Sequential means layers are stacked in order, top to bottom.
    model = tf.keras.Sequential(
        [
            # Input layer: each sample has exactly 2 numbers (x1 and x2).
            tf.keras.layers.Input(shape=(2,)),

            # Hidden layer 1:
            # 8 neurons, ReLU activation.
            # ReLU lets the model learn non-linear patterns.
            tf.keras.layers.Dense(8, activation="relu"),

            # Hidden layer 2:
            # another 8-neuron layer for extra learning capacity.
            tf.keras.layers.Dense(8, activation="relu"),

            # Output layer:
            # 1 neuron with sigmoid activation outputs a probability 0..1
            # for binary classification.
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile chooses how training will work:
    # - optimizer="adam": updates weights efficiently
    # - loss="binary_crossentropy": good for 0/1 classification
    # - metrics=["accuracy"]: shows percent correct during training/eval
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # Set random seeds so you get similar results each run.
    # Useful for learning/debugging because output is more repeatable.
    np.random.seed(42)
    tf.random.set_seed(42)

    # Make training data (what the model learns from).
    x_train, y_train = make_data(1500)

    # Make separate test data (used only after training).
    # This helps measure how well the model generalizes.
    x_test, y_test = make_data(300)

    # Build an untrained neural network model.
    model = build_model()

    # Train the model:
    # - epochs=10 means the model sees the training set 10 times
    # - batch_size=32 means it learns from 32 samples at a time
    # - verbose=1 prints training progress each epoch
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Evaluate on test data.
    # loss is the error value; accuracy is percent correct.
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    # Try a few custom points to see predictions.
    # Each row is [x1, x2].
    samples = np.array([[1.0, 1.0], [-1.0, -0.5], [0.2, -0.1]], dtype=np.float32)

    # Predict probabilities for each sample.
    # Example: 0.92 means "92% likely class 1".
    probs = model.predict(samples, verbose=0).flatten()

    # Convert probabilities into hard class labels:
    # > 0.5 -> 1, otherwise -> 0.
    preds = (probs > 0.5).astype(int)

    # Print both probabilities and final class predictions.
    print("Sample probabilities:", np.round(probs, 3).tolist())
    print("Sample predictions:", preds.tolist())


# This standard Python pattern means:
# run main() only if this file is executed directly.
# (If imported from another file, main() will not auto-run.)
if __name__ == "__main__":
    main()
