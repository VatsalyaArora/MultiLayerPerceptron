import numpy as np

class MSELoss:
    def calculate(self, prediction, target):
        return np.mean(np.power(prediction - target, 2))
    
    def backward(self, prediction, target):
        n_samples = prediction.shape[0]
        return 2 * (prediction - target) / n_samples