# ==========================================
# Compare Keras vs PyTorch
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual results
keras_accuracy = 0.9825
pytorch_accuracy = 0.9808

comparison = pd.DataFrame({
    "Framework": [
        "TensorFlow / Keras",
        "PyTorch"
    ],
    "Accuracy": [
        keras_accuracy,
        pytorch_accuracy
    ]
})

print(comparison)

plt.figure(figsize=(6,5))

plt.bar(
    comparison["Framework"],
    comparison["Accuracy"]
)

plt.title("Keras vs PyTorch Accuracy")
plt.ylabel("Accuracy")

for i, value in enumerate(comparison["Accuracy"]):
    plt.text(i, value + 0.002, f"{value:.4f}", ha="center")

plt.show()
