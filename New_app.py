import os
import time
import requests
import streamlit as st
from typing import Any
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from transformers import pipeline

import torch
import torchaudio
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# CSS for UI Styling
css_code = """
<style>
    section[data-testid="stSidebar"] > div > div:nth-child(2) {
        padding-top: 0.75rem !important;
    }
    section.main > div {
        padding-top: 64px;
    }
</style>
"""

# Progress Bar
def progress_bar(duration: int) -> None:
    progress_text = "Processing..."
    my_bar = st.progress(0, text=progress_text)
    for i in range(duration):
        time.sleep(0.02)
        my_bar.progress(i + 1, text=progress_text)
    my_bar.empty()

# Generate Caption from Image
def generate_text_from_image(image_path: str) -> str:
    caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = caption_pipeline(image_path)[0]["generated_text"]
    return caption

# Generate Story from Caption
def generate_story_from_text(scenario: str) -> str:
    prompt_template = """
    You are a creative storyteller. Create a short, engaging, and imaginative story (max 50 words) using the following context:

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["scenario"])
    llm = ChatGroq(model_name="llama-3.2-11b-vision-preview", temperature=0.9)
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)
    
    return story_llm.predict(scenario=scenario)

def generate_speech_from_text(message: str):
    model_name = "SparkAudio/Spark-TTS-0.5B"
    processor = SpeechT5Processor.from_pretrained(model_name)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
    if not message.strip():
        st.error("No text received for speech generation.")
        return
    
    with torch.no_grad():
        inputs = processor(message, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output_waveform = model.generate(**inputs)
    
    # Convert tensor to audio
    audio_path = "generated_audio.wav"
    torchaudio.save(audio_path, output_waveform.cpu(), 22050)  # Save with a sample rate of 22.05kHz
    
    st.success("Speech generation successful!")
    st.audio(audio_path)  # Play the generated audio in Streamlit


# Streamlit UI
def main():
    st.set_page_config(page_title="AI Image to Story Generator", page_icon="📖")
    st.markdown(css_code, unsafe_allow_html=True)
    
    st.title("🖼️ AI Image to Story Generator 📖")
    st.sidebar.write("Created by **PragyanAI** - Education Purpose")

    uploaded_file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "png"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with open(uploaded_file.name, "wb") as file:
            file.write(uploaded_file.getvalue())

        progress_bar(100)
        
        scenario = generate_text_from_image(uploaded_file.name)
        story = generate_story_from_text(scenario)
        generate_speech_from_text(story)

        with st.expander("Generated Image Description"):
            st.write(scenario)

        with st.expander("Generated Story"):
            st.write(story)

        st.audio("generated_audio.flac")

if __name__ == "__main__":
    main()
