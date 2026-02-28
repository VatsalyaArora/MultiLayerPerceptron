import numpy as np

class NestedLayers:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer, limit_type = 'relu'):
        layer.limit_type = limit_type
        self.layers.append(layer)

    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            current_input = layer.weight_calculation(current_input)
        return current_input
    
    def backward(self, error):
        current_error = error
        for layer in reversed(self.layers):
            current_error = layer.error_calculation(current_error)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.error_implementation(learning_rate)
    
    def save_model(self, folder = "saved_model"):
        for i, layer in enumerate(self.layers):
            layer.save_layer(folder, i)
        print(f"Model saved to folder: {folder}")

    def load_model(self, folder = "saved_model"):   
        for i, layer in enumerate(self.layers):
            layer.load_layer(folder, i)
        print(f"Model loaded from folder: {folder}")        