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

    data = np.loadtxt("Data/ToPredict.csv", delimiter=",", skiprows=1)
    X = data[:, :10]          
    
    risk_score = app_model.forward(X).ravel()
    
    for i in range(5):
        print(f"Predicted Risk Score: {risk_score[i]:.4f}")
        if risk_score[i] > 0.5:
            print("Status: HIGH RISK - Manual Review Required")
        else:
            print("Status: LOW RISK - Auto-Approved")

if __name__ == "__main__":
    run_prediction()
    
    