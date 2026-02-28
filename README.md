Multi-Layer Perceptron (MLP) from Scratch
A pure NumPy implementation of a Neural Network engine, developed as part of my Master's studies in Artificial Intelligence at the University of Milan-Bicocca. This project demonstrates the core mechanics of deep learning—backpropagation, weight management, and gradient descent—without relying on high-level libraries like PyTorch or TensorFlow.

🚀 Project Overview
This engine is applied to a Financial Risk Prediction use case, where it evaluates 10 key normalized features to determine loan eligibility.

Goal: Classify candidates as "Approved" (0.0) or "Denied" (1.0) based on financial risk.

Architecture: Flexible SequentialChain allowing for custom layers and activation functions.

Performance: Achieved a loss reduction from ~0.39 to ~0.04 during training.

🛠️ Technical Features
Custom Backpropagation: Manual implementation of the chain rule and gradient updates.

Activation Functions: Support for ReLU (hidden layers) and Sigmoid (output layer).

Weight Management: Capability to save and load trained model states via .npy files.

Loss Function: Mean Squared Error (MSE) for regression-to-classification mapping.

📂 Repository Structure
/Core: The mathematical heart of the engine (Engine, WeightMatrix, Losses).

/Examples: Training scripts and data generation logic.

/Application: Real-world implementation (Predictor).

/finance_model: Pre-trained weights for immediate inference.

🏃 How to Run
Train the Model:

Bash

python -m Examples.TrainModel
Run Inference:

Bash

python Application/Predictor.py
Run Unit Tests:

Bash

python unittesting.py
