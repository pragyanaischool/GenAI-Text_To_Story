import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatGroq  # Using ChatGroq instead of ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline

from utils.custom import css_code

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def progress_bar(amount_of_time: int) -> Any:
    progress_text = "Please wait, Generative models hard at work"
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

def generate_text_from_image(url: str) -> str:
    image_to_text: Any = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    generated_text: str = image_to_text(url)[0]["generated_text"]
    print(f"IMAGE INPUT: {url}")
    print(f"GENERATED TEXT OUTPUT: {generated_text}")
    return generated_text

def generate_story_from_text(scenario: str) -> str:
    prompt_template: str = f"""
    You are a talented storyteller who can create a story from a simple narrative.
    Create a story using the following scenario; the story should be a maximum of 50 words long;
    
    CONTEXT: {scenario}
    STORY:
    """
    prompt: PromptTemplate = PromptTemplate(template=prompt_template, input_variables=["scenario"])
    llm: Any = ChatGroq(model_name="llama-2-13b-chat", temperature=0.9)  # Using Groq's Llama model
    story_llm: Any = LLMChain(llm=llm, prompt=prompt, verbose=True)
    generated_story: str = story_llm.predict(scenario=scenario)
    print(f"TEXT INPUT: {scenario}")
    print(f"GENERATED STORY OUTPUT: {generated_story}")
    return generated_story

def generate_speech_from_text(message: str) -> Any:
    API_URL: str = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers: dict[str, str] = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads: dict[str, str] = {"inputs": message}
    response: Any = requests.post(API_URL, headers=headers, json=payloads)
    with open("generated_audio.flac", "wb") as file:
        file.write(response.content)

def main() -> None:
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)
    st.image("PragyanAI_Transperent_github.png
    with st.sidebar:
        st.write("AI App created by @ PragyanAI - Education Purpose")
    st.header("Image-to-Story Converter")
    uploaded_file: Any = st.file_uploader("Please choose a file to upload", type="jpg")
    if uploaded_file is not None:
        bytes_data: Any = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        progress_bar(100)
        scenario: str = generate_text_from_image(uploaded_file.name)
        story: str = generate_story_from_text(scenario)
        generate_speech_from_text(story)
        with st.expander("Generated Image scenario"):
            st.write(scenario)
        with st.expander("Generated short story"):
            st.write(story)
        st.audio("generated_audio.flac")

if __name__ == "__main__":
    main()
