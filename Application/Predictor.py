import numpy as np
from Core.Engine import NestedLayers
from Core.WeightMatrix import WeightMat

def run_prediction():
    
    app_model = NestedLayers()
    app_model.add_layer(WeightMat(10, 7),   limit_type = 'relu')
    app_model.add_layer(WeightMat(7, 4), limit_type = 'relu')
    app_model.add_layer(WeightMat(4, 1), limit_type = 'sigmoid')

    try:
        app_model.load_model("finance_model")
    except FileNotFoundError:
        print("Model not found. Please train the model first using TrainModel.py.")
        return
    
    print("\n--- Running Prediction ---")

    user_data = np.array([[
        0.9, # Income Level (normalized) 0.05
        0.9, # Credit Score (normalized) 0.2
        0.9, # Debt-to-Income Ratio (normalized) 0.3
        0.6, # Employment Stability (normalized) 0.03
        0.4, # Age (normalized) 0.01
        0.6, # Savings (normalized) 0.1 
        0.1, # Open Loans (normalized) 0.1
        0.5, # Payment History (normalized) 0.05
        0.1, # Requested Amount (normalized) 0.15
        1.0  # Residence 0.01
    ]])

    predicton = app_model.forward(user_data)
    risk_score = predicton[0, 0]

    print(f"Predicted Risk Score: {risk_score:.4f}")
    if risk_score > 0.5:
        print("Status: HIGH RISK - Manual Review Required")
    else:
        print("Status: LOW RISK - Auto-Approved")
    print(app_model.forward(np.ones((1, 10))))

if __name__ == "__main__":
    run_prediction()
    
    