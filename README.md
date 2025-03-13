An app that uses Hugging Face AI models to generate text from an image, which then generates audio from the text.

Execution is divided into 3 parts:

1. Image to text: an image-to-text transformer model (Salesforce/blip-image-captioning-base) is used to generate a text scenario based on the on the AI understanding of the image context

2. Text to story: OpenAI LLM model is prompted to create a short story (50 words: can be adjusted as reqd.) based on the generated scenario. gpt-3.5-turbo

3. Story to speech: a text-to-speech transformer model (espnet/kan-bayashi_ljspeech_vits) is used to convert the generated short story into a voice-narrated audio file

A user interface is built using streamlit to enable uploading the image and playing the audio file
