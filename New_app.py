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
def generate_speech_from_text(message: str) -> None:
    API_URL = "https://api-inference.huggingface.co/models/mrfakename/SparkAudio-Spark-TTS-0.5B"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}", "Content-Type": "application/json"}

    if not message.strip():
        st.error("No text received for speech generation.")
        return

    payload = {
        "text": message,
        "language": "en",  # Set language (modify as needed)
        "speaker": "default",  # Change speaker if applicable
        "speed": 1.0,  # Adjust speed (1.0 = normal, <1 = slower, >1 = faster)
        "format": "wav"  # Ensure output format is correct
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        audio_data = response.content  # Get binary audio data
        with open("generated_audio.wav", "wb") as file:
            file.write(audio_data)  # Save audio file

        st.success("Speech generation successful!")
        st.audio("generated_audio.wav")  # Play the audio in Streamlit
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating speech: {e}")

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
