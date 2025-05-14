import torch  # PyTorch for loading model
import streamlit as st  # Streamlit for building UI
import json  # For reading/writing JSON files
import os  # For file system operations
from pathlib import Path  # For path manipulations
import pyttsx3  # For text-to-speech
from train import train_model  # Training logic
from chatbot import load_model, get_response  # Chatbot logic

# Set Streamlit app title
st.title("🧠 NLP Chatbot Trainer & Assistant")

# Get the path to the saved model
model_path = Path(__file__).resolve().parent / "trained_model" / "trained_model.pth"
sample_intents_path = Path(__file__).resolve().parent / "sample_data" / "intents.json"

# Set initial state variables if not already set
if "model_trained" not in st.session_state:
    st.session_state.model_trained = os.path.exists(model_path)
if "intents_data" not in st.session_state:
    st.session_state.intents_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Section to upload a custom intents.json file
uploaded_file = st.file_uploader("📄 Upload your intents.json file", type=["json"])

# Checkbox to use default sample intents instead of uploading
use_default = st.checkbox("📂 Use default intents.json from sample_data")

# Handle default intents.json checkbox
if use_default:
    with open(sample_intents_path, "r") as f:
        intents = json.load(f)
    st.session_state.intents_data = intents
    st.success("✅ Loaded default intents.json")

# If a user uploads a file manually
elif uploaded_file is not None:
    intents = json.load(uploaded_file)
    # Save uploaded file locally
    with open(sample_intents_path, "w") as f:
        json.dump(intents, f)
    st.session_state.intents_data = intents
    st.success("✅ Uploaded and saved intents.json")

# Epochs input for training
epochs = st.number_input("🛠️ Training Epochs", min_value=100, max_value=10000, value=1000, step=100)

# Train button
if st.session_state.intents_data and st.button("🚀 Train Chatbot"):
    train_model(st.session_state.intents_data, num_epochs=epochs)
    st.session_state.model_trained = True
    st.success("✅ Training complete! You can now chat with your bot.")

# If model is trained, show chat interface
if st.session_state.model_trained:
    st.subheader("💬 Chat with your bot")

    # Text input for the user
    user_input = st.text_input("You:", "")

    # Button to send message
    if st.button("Send") and user_input:
        # Load model and metadata
        model, data = load_model()
        # Get chatbot response
        response = get_response(
            model,
            st.session_state.intents_data["intents"],
            user_input,
            data["all_words"],
            data["tags"]
        )
        # Append chat to history
        st.session_state.chat_history.append((user_input, response))
        st.rerun()  # Refresh UI to show latest chat

    # Display previous chat history
    for user_msg, bot_reply in st.session_state.chat_history:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**🤖 Bot:** {bot_reply}")

    # Add button to convert bot's last response to speech
    if st.session_state.chat_history:
        if st.button("🔊 Speak Bot's Last Response"):
            engine = pyttsx3.init()
            last_response = st.session_state.chat_history[-1][1]
            engine.say(last_response)
            engine.runAndWait()
else:
    st.info("Please train the chatbot or load default intents to start chatting.")
