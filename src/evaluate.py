# ==========================================
# Evaluate Saved Keras Model
# ==========================================

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test / 255.0

# Load Model
model = load_model("../models/keras_model.keras")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Keras Test Accuracy: {accuracy:.4f}")
