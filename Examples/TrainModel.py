import numpy as np
import pandas as pd
from Core.WeightMatrix import WeightMat
from Core.Engine import NestedLayers
from Core.Losses import MSELoss

seed = 42

data = np.loadtxt("Data/data.csv", delimiter=",", skiprows=1)
X = data[:, :10]
Y = data[:, 11:12] 

rng = np.random.default_rng(seed)
idx = rng.permutation(len(X))
train, test = idx[:8000], idx[8000:]
X_train, Y_train = X[train], Y[train]
X_test,  Y_test  = X[test],  Y[test]

learning_rate = 0.5

np.random.seed(0)

model = NestedLayers()
model.add_layer(WeightMat(10, 7),   limit_type = 'relu')
model.add_layer(WeightMat(7, 4), limit_type = 'relu')
model.add_layer(WeightMat(4, 1), limit_type = 'sigmoid')

criterion = MSELoss()

for epoch in range(5000):
    prediction = model.forward(X_train)

    loss =  criterion.calculate(prediction, Y_train)
    error = criterion.backward(prediction, Y_train)

    model.backward(error)

    model.update_weights(learning_rate)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

test_pred = (model.forward(X_test) > 0.5).astype(int)
print("Test accuracy:", (test_pred == Y_test).mean())

print(model.forward(np.ones((1, 10))))
model.save_model("finance_model")