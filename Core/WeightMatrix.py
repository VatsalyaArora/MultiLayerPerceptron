import numpy as np
import os

class WeightMat:
    def __init__(self, n_inputs, n_neurons, limit_type = 'relu'):
        self.weights = np.random.randn(n_inputs , n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1 , n_neurons))
        
        self.limit_type = limit_type
        
        self.input_cache = None
        self.z_cache = None
        self.output_cache = None

        #print(f"[INIT] Created WeightMatrix: {n_inputs} inputs -> {n_neurons} neurons")
    
    def weight_calculation(self, inputs):
        self.input_cache = inputs
        self.z_cache = np.dot(inputs, self.weights) + self.biases
        self.output_cache = self.threshold_limit(self.z_cache)

        #print(f"\n[FORWARD] Input Shape: {inputs.shape} | Output Shape: {self.output_cache.shape}")
        #print(f"        Avg Signal Strength: {np.mean(self.output_cache):.4f}")

        return self.output_cache
        
    def threshold_limit(self, z):
        if self.limit_type == 'relu' :
            return np.maximum(0,z)
        elif self.limit_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        return z
        
    def error_calculation(self, back_error):
        back_z = back_error.copy()
        if self.limit_type == 'relu':
            back_z[self.z_cache <= 0] = 0
        elif self.limit_type == 'sigmoid':
            sigmoid_output = 1 / (1 + np.exp(-self.z_cache))
            back_z *= sigmoid_output * (1 - sigmoid_output)
            
        self.back_weights = np.dot(self.input_cache.T, back_z)
        self.back_biases = np.sum(back_z, axis = 0, keepdims = True)
        
        back_input = np.dot(back_z, self.weights.T)
        
        #print(f"[BACKWARD] Incoming Error Mean: {np.mean(back_z):.6f}")
        #print(f"           Weight Gradient (sum): {np.sum(self.back_weights):.6f}")
        #print(f"           Passing back Error Shape: {back_input.shape}")

        return back_input
        
    def error_implementation(self, learning_rate):
        #old_w_mean = np.mean(self.weights)
        self.weights -= learning_rate * self.back_weights
        self.biases -= learning_rate * self.back_biases

        #print(f"[UPDATE] Weights adjusted. Mean shift: {np.mean(self.weights) - old_w_mean:.8f}")

    def save_layer(self, folder, layer_id):
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(f"{folder}/w_{layer_id}.npy", self.weights)
        np.save(f"{folder}/b_{layer_id}.npy", self.biases)
     
    def load_layer(self, folder, layer_id):
        self.weights = np.load(f"{folder}/w_{layer_id}.npy")
        self.biases = np.load(f"{folder}/b_{layer_id}.npy") 