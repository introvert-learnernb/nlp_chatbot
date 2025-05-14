import torch  # PyTorch for model and tensor operations
import random  # To randomly choose a response
from model import NeuralNet  # Import the trained model class
from utils import bag_of_words, tokenize  # Helper functions
from pathlib import Path  # For file path operations


# Function to load the trained model and metadata
def load_model(model_path= Path(__file__).resolve().parent / "trained_model" / "trained_model.pth"):
    data = torch.load(model_path)  # Load model and data from file
    
    # Rebuild model using saved parameters
    model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
    model.load_state_dict(data["model_state"])  # Load trained weights
    model.eval()  # Set model to evaluation mode (no training)

    return model, data  # Return model and metadata like all_words, tags

# Function to generate a response based on user input
def get_response(model, intents, user_input, all_words, tags, threshold=0.5):
    sentence = tokenize(user_input)  # Tokenize user input
    X = bag_of_words(sentence, all_words)  # Convert input to bag-of-words format
    X = torch.from_numpy(X).float().unsqueeze(0)  # Convert to tensor and add batch dimension

    output = model(X)  # Get output predictions from model
    _, predicted = torch.max(output, dim=1)  # Get the index of the highest score
    tag = tags[predicted.item()]  # Map index to tag name

    probs = torch.softmax(output, dim=1)  # Convert scores to probabilities
    prob = probs[0][predicted.item()]  # Get the probability of the predicted tag

    if prob.item() > threshold:  # If above confidence threshold
        for intent in intents:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])  # Return a random response for the matched tag

    return "🤖 Sorry, I didn't understand that."

