import torch.nn as nn


#Neural Network Model
class NeuralNet(nn.Module):  # Define a class NeuralNet that inherits from nn.Module (base class for all neural network modules)
    def __init__(self,input_size, hidden_size, output_size):  # Constructor with input, hidden, and output layer sizes
        super(NeuralNet, self).__init__()  # Initialize the parent nn.Module class
        self.l1 = nn.Linear(input_size, hidden_size)  # First linear layer: input to hidden
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Second linear layer: hidden to hidden
        self.l3 = nn.Linear(hidden_size, output_size)  # Third linear layer: hidden to output
        self.relu = nn.ReLU()  # ReLU activation function
        
    def forward(self, x):  # Forward pass method, defines how data flows through the network
        out = self.l1(x)  # Pass input through first linear layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.l2(out)  # Pass through second linear layer
        out = self.relu(out)  # Apply ReLU activation again
        out = self.l3(out)  # Pass through third linear layer
        return out  # Return the final output
    