# MNIST Handwritten Digit Recognition Using Keras and PyTorch

## Overview

This project demonstrates handwritten digit recognition using the MNIST dataset. The same Deep Neural Network architecture was implemented using both TensorFlow Keras and PyTorch to understand the similarities and differences between these two popular Deep Learning frameworks.

The project covers data preprocessing, model training, evaluation, visualization, framework comparison, and analysis of misclassified handwritten digits.

---

## Project Objectives

- Load the MNIST dataset.
- Normalize image pixel values.
- Build a Deep Neural Network using TensorFlow Keras.
- Build the same Neural Network using PyTorch.
- Train and evaluate both models.
- Compare the performance of Keras and PyTorch.
- Visualize training accuracy and loss.
- Analyze incorrectly classified digits.

---

## Dataset

**Dataset Name:** MNIST Handwritten Digits

| Feature | Value |
|----------|-------|
| Total Images | 70,000 |
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 × 28 Pixels |
| Classes | 10 (Digits 0–9) |

The dataset is automatically loaded using:

```python
from tensorflow.keras.datasets import mnist
```

No manual download is required.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Google Colab

---

## Neural Network Architecture

```
Input Image (28 × 28)

↓

Flatten Layer

↓

Dense Layer (128 Neurons)

↓

ReLU Activation

↓

Dense Layer (64 Neurons)

↓

ReLU Activation

↓

Output Layer (10 Neurons)

↓

Softmax Activation
```

---

## Project Workflow

1. Import Required Libraries
2. Load the MNIST Dataset
3. Normalize Image Pixel Values
4. Build the Neural Network in TensorFlow Keras
5. Train and Evaluate the Keras Model
6. Build the Same Neural Network in PyTorch
7. Train and Evaluate the PyTorch Model
8. Compare Model Performance
9. Visualize Training Results
10. Analyze Wrong Predictions

---

## Results

| Framework | Test Accuracy |
|-----------|---------------|
| TensorFlow / Keras | XX.XX% |
| PyTorch | XX.XX% |

> Replace the above values with your actual model accuracies.

---

## Visualizations

The project includes the following visualizations:

- Sample MNIST Images
- Keras Training Accuracy
- Keras Training Loss
- PyTorch Training Accuracy
- Keras vs PyTorch Accuracy Comparison
- Wrong Prediction Analysis

---

## Project Structure

```
MNIST-Handwritten-Digit-Recognition-Using-Keras-and-PyTorch
│
├── data
│   └── README.md
│

│
├── models
│   ├── keras_model.keras
│   └── pytorch_model.pth
│
├── notebook
│   └── MNIST_Handwritten_Digit_Recognition.ipynb
│
├── src
│   ├── keras_train.py
│   ├── pytorch_train.py
│   ├── evaluate.py
│   └── compare_models.py
│
├── README.md
├── requirements.txt
├── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YourGitHubUsername/MNIST-Handwritten-Digit-Recognition-Using-Keras-and-PyTorch.git
```

Navigate to the project directory:

```bash
cd MNIST-Handwritten-Digit-Recognition-Using-Keras-and-PyTorch
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the Keras implementation:

```bash
python src/keras_train.py
```

Run the PyTorch implementation:

```bash
python src/pytorch_train.py
```

Evaluate the saved model:

```bash
python src/evaluate.py
```

Compare both models:

```bash
python src/compare_models.py
```

---

## Key Learning Outcomes

- Deep Learning Fundamentals
- Artificial Neural Networks
- TensorFlow Keras
- PyTorch
- Image Classification
- Data Preprocessing
- Model Evaluation
- Framework Comparison
- Performance Visualization
- Error Analysis

---

## Future Improvements

- Implement Convolutional Neural Networks (CNN)
- Hyperparameter Tuning
- Data Augmentation
- Deploy as a Streamlit Application
- Deploy using Flask
- Experiment with Advanced Deep Learning Architectures

---

## Author

**Soha Ayub**

Software Engineering Student

AI & Machine Learning Enthusiast

AI Internship Project – Cloulem

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- TensorFlow
- PyTorch
- MNIST Dataset
- Google Colab
