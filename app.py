import torch
import streamlit as st
import json
import os
from pathlib import Path
from gtts import gTTS
import tempfile

from train import train_model
from chatbot import load_model, get_response

st.title("🧠 NLP Chatbot Trainer & Assistant")

model_path = Path(__file__).resolve().parent / "trained_model" / "trained_model.pth"
sample_intents_path = Path(__file__).resolve().parent / "sample_data" / "intents.json"

if "model_trained" not in st.session_state:
    st.session_state.model_trained = os.path.exists(model_path)
if "intents_data" not in st.session_state:
    st.session_state.intents_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("📄 Upload your intents.json file", type=["json"])
use_default = st.checkbox("📂 Use default intents.json")

if use_default:
    with open(sample_intents_path, "r") as f:
        intents = json.load(f)
    st.session_state.intents_data = intents
    st.success("✅ Loaded default intents.json")

elif uploaded_file is not None:
    intents = json.load(uploaded_file)
    with open(sample_intents_path, "w") as f:
        json.dump(intents, f)
    st.session_state.intents_data = intents
    st.success("✅ Uploaded and saved intents.json")

epochs = st.number_input("🛠️ Training Epochs", min_value=100, max_value=10000, value=1000, step=100)

if st.session_state.intents_data and st.button("🚀 Train Chatbot"):
    train_model(st.session_state.intents_data, num_epochs=epochs)
    st.session_state.model_trained = True
    st.success("✅ Training complete! You can now chat with your bot.")

if st.session_state.model_trained:
    st.subheader("💬 Chat with your bot")

    user_input = st.text_input("You:", "")

    if st.button("Send") and user_input:
        model, data = load_model()
        response = get_response(
            model,
            st.session_state.intents_data["intents"],
            user_input,
            data["all_words"],
            data["tags"]
        )
        st.session_state.chat_history.append((user_input, response))
        st.rerun()

    for user_msg, bot_reply in st.session_state.chat_history:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**🤖 Bot:** {bot_reply}")

    # ✅ Use gTTS for last response audio
    if st.session_state.chat_history:
        if st.button("🔊 Speak Bot's Last Response"):
            last_response = st.session_state.chat_history[-1][1]
            tts = gTTS(text=last_response, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                st.audio(tmp_file.name, format="audio/mp3")
else:
    st.info("Please train the chatbot or load default intents to start chatting.")
