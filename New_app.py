import os
import time
import asyncio
from typing import Any

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# CSS for UI improvements
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

def progress_bar(amount_of_time: int) -> None:
    """Displays a progress bar in Streamlit."""
    progress_text = "Please wait, Generative models are working..."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    
    time.sleep(1)
    my_bar.empty()

def generate_text_from_image(url: str) -> str:
    """Generates a caption from an image using a Hugging Face model."""
    try:
        image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0)
        generated_text = image_to_text(url)[0]["generated_text"]
        return generated_text
    except Exception as e:
        st.error(f"Error generating text from image: {e}")
        return ""

async def generate_story_from_text(scenario: str) -> str:
    """Generates a short story from the given scenario using Groq's Llama model."""
    try:
        prompt = PromptTemplate(
            template="""You are a talented storyteller. Create a story from the scenario given below. 
            The story should be a maximum of 50 words long. 

            CONTEXT: {scenario} 
            STORY: """,
            input_variables=["scenario"]
        )

        llm = ChatGroq(model_name="llama-3.2-11b-vision-preview", temperature=0.9)
        story_chain = prompt | llm  # Modern LangChain usage
        
        generated_story = await story_chain.ainvoke({"scenario": scenario})
        return generated_story
    except Exception as e:
        st.error(f"Error generating story: {e}")
        return ""

def generate_speech_from_text(message: str) -> None:
    """Generates speech from text using Hugging Face's text-to-speech model."""
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    if not HUGGINGFACE_API_TOKEN:
        st.error("Hugging Face API token is missing! Set it in the environment variables.")
        return

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": message})
        response.raise_for_status()  # Raise an error for bad responses
        
        with open("generated_audio.flac", "wb") as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating speech: {e}")

def run_async_task(coro):
    """Runs an async function safely in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def main() -> None:
    """Main function for the Streamlit app."""
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)
    
    st.image("PragyanAI_Transperent_github.png")
    
    with st.sidebar:
        st.write("AI App created by @ PragyanAI - Education Purpose")
    
    st.header("Image-to-Story Converter")
    
    uploaded_file = st.file_uploader("Please choose a file to upload", type=["jpg", "png"])
    
    if uploaded_file:
        file_path = f"temp_{uploaded_file.name}"
        
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getvalue())
        
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        progress_bar(100)
        
        scenario = generate_text_from_image(file_path)
        
        if scenario:
            story = run_async_task(generate_story_from_text(scenario))
            generate_speech_from_text(story)
            
            with st.expander("Generated Image Scenario"):
                st.write(scenario)
            
            with st.expander("Generated Short Story"):
                st.write(story)
            
            st.audio("generated_audio.flac")

if __name__ == "__main__":
    main()
