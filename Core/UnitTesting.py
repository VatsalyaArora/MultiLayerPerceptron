import numpy as np
from WeightMatrix import WeightMat
from Engine import NestedLayers

model = NestedLayers()
model.add_layer(WeightMat(10, 7),   limit_type = 'relu')
model.add_layer(WeightMat(7, 4), limit_type = 'relu')
model.add_layer(WeightMat(4, 1), limit_type = 'sigmoid')

model.load_model("..\\finance_model")

goodTest = np.array([[0.9, 0.9, 0.1, 0.8, 0.5, 0.9, 0.1, 0.9, 0.1, 1.0]])
goodPrediction = model.forward(goodTest)

badTest = np.array([[0.1, 0.2, 0.9, 0.2, 0.3, 0.1, 0.8, 0.3, 0.9, 0.0]])
badPrediction = model.forward(badTest)

print(f"Good Candidate Score (Target ~0): {goodPrediction[0][0]:.4f}")
print(f"Bad Candidate Score (Target ~1): {badPrediction[0][0]:.4f}")

#print("\nTarget Weight for first input: 2.0")
#print(f"Actual Weight Learned: {layer.weights[0][0]:.4f}")