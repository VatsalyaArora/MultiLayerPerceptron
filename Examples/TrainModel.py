import numpy as np
from Core.WeightMatrix import WeightMat
from Core.Engine import NestedLayers
from Core.Losses import MSELoss

# --- 1. GENERATE LABELED DATA ---
def generate_labeled_data(samples=100):
    X = []
    Y = []
    for _ in range(samples // 2):
        # Set A: "Approved" (Target 0) - High Income/Credit, Low Debt
        good_profile = np.random.uniform(0.7, 1.0, 10) 
        good_profile[2] = np.random.uniform(0.0, 0.05) # Force Low Debt
        X.append(good_profile)
        Y.append([0.0])
        
        print(f"Good Profile Value: {good_profile} | good_profile[2] (Debt): {good_profile[2]:.4f}")

        # Set B: "Denied" (Target 1) - Low Income/Credit, High Debt
        bad_profile = np.random.uniform(0.0, 0.4, 10)
        bad_profile[2] = np.random.uniform(0.95, 1.0) # Force High Debt
        X.append(bad_profile)
        Y.append([1.0])


    
    return np.array(X), np.array(Y)

X_train, Y_train = generate_labeled_data(200)

learning_rate = 0.05

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

print(model.forward(np.ones((1, 10))))
model.save_model("finance_model")