import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from utils import tokenize, stem, bag_of_words, preprocess_intents
from model import NeuralNet
from pathlib import Path
import streamlit as st

# Create a custom dataset class
class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.x_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def train_model(json_data, num_epochs=1000, batch_size=8, learning_rate=0.001, hidden_size=512):
    print("Starting data preprocessing...")
    X_train, y_train, all_words, tags = preprocess_intents(json_data)
    
    # Convert X_train, y_train, all_words, and tags to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    all_words = np.array(all_words)
    tags = np.array(tags)
    
    print(f"Data preprocessing complete. Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")

    input_size = X_train.shape[1]  # Number of input features (length of bag-of-words vector)
    output_size = len(tags)  # Number of output classes (unique tags)

    # Create dataset object
    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if CUDA (GPU) is available
    model = NeuralNet(input_size, hidden_size, output_size).to(device)  # Initialize model

    criterion = nn.CrossEntropyLoss()  # Define loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Define optimizer

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training loop
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)  # Move input to GPU if available
            labels = labels.to(dtype=torch.long).to(device)  # Move labels to GPU

            # Forward pass
            outputs = model(words)  # Predict the output using the model
            loss = criterion(outputs, labels)  # Calculate the loss

            optimizer.zero_grad()  # Zero out the gradients before backward pass
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model weights

        
        
        # Update progress bar and status text in Streamlit
        percentage = int((epoch + 1) / num_epochs * 100)
        progress_bar.progress(percentage)
        status_text.text(
            f"Training... Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f}"
        )

    print(f'Final Loss: {loss.item():.4f}')  # Final loss after training

    # Save model and metadata
    data = {
        "model_state": model.state_dict(),  # Save the trained model weights
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words.tolist(),  # Convert to list before saving
        "tags": tags.tolist()  # Convert to list before saving
    }

    FILE = Path(__file__).resolve().parent / "trained_model" / "trained_model.pth"  # Path to save the model
    torch.save(data, FILE)  # Save the model and data

    print(f'Training complete. File saved to {FILE}')
    return FILE



def load_intents(file_path):
        with open(file_path, 'r') as f:
            return json.load(f) 
        
        
# Main execution to call the function
if __name__ == "__main__":
    # Load the intents.json file from the sample_data folder
    file_path = Path(__file__).resolve().parent / "sample_data" / "intents.json"

    intents_data = load_intents(file_path)  # Correctly load the JSON data into a dictionary
    # The training function
    # Call the training function with the loaded data
    trained_model_path = train_model(intents_data)

    print(f'Model saved to: {trained_model_path}')
