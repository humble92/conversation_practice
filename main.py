# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PIL import UnidentifiedImageError
from dotenv import find_dotenv, load_dotenv
import logging
import requests
import streamlit as st
import os

load_dotenv(find_dotenv())

# env. variables
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")
API_BASE_URL = 'https://api-inference.huggingface.co/models/'
headers = {"Authorization": f"Bearer {HF_HUB_TOKEN}"}


def home():
    st.title("AI dictionary on your fingertip")
    st.write('Expand your English skills with the help of AI. '
             'AI generates conversation scenarios to help you practice and provides you with useful expressions. '
             'Put close the AI tutor at your fingertip and jump up to the next leve. Have fun!')

    st.subheader("Context", divider='rainbow')
    style = st.selectbox(
        'In which context are you talking?',
        ('Daily life', 'Business', 'Academic', 'Social', 'Technical', 'Medical', 'Art'))
    st.write('You selected:', style)

    st.subheader("Describe a situation.", divider='rainbow')
    tab1, tab2 = st.tabs(["🖼️ Image", "🖹 Text"])

    # print_hi('hugging couples in the desert')

    with tab1:
        st.write('Upload a picture showing the conversation context. '
                 'AI helps you practice English conversations in the specific scenario.')

        generate_scenario_by_picture(style)

    with tab2:
        st.write('Describe a situation explaining the conversation context. '
                 'AI helps you practice English conversations in the specific scenario.')

        generate_scenario_by_text(style)


def generate_scenario_by_text(style):
    st.write("Context detail")
    input_text = st.text_area("Describe a situation you want to practice English conversation.")
    if st.button("Generate", type="secondary"):
        with st.spinner('Thinking and processing...'):
            # Generate a picture book
            conversation = generating_scenario(input_text, style)

            try:
                conversation_image = text2image(conversation)
                st.image(conversation_image, caption='Generated Image.', use_column_width=True)
            except UnidentifiedImageError:
                # When it fails to draw pictures, just skip and go ahead
                print("Temporary failure to draw pictures....")
                pass

            # Generate an audio book
            audio_bytes = text2speech(conversation)

            with st.expander("Conversation Scenario"):
                st.write(conversation)

            st.audio(audio_bytes)


def generate_scenario_by_picture(style):
    uploaded_picture = st.file_uploader("Upload a picture showing the conversation context.", type=['png', 'jpg'])

    if uploaded_picture:
        logging.info(uploaded_picture)

        bytes_data = uploaded_picture.getvalue()
        st.image(uploaded_picture, caption='Uploaded Image.', use_column_width=True)

        with st.spinner('Thinking and processing...'):
            full_path = os.path.join('uploads', uploaded_picture.name)
            with open(full_path, "wb") as file:
                file.write(bytes_data)

            image_text = img2txt(full_path)
            conversation = generating_scenario(image_text, style)

            audio_bytes = text2speech(conversation)

            with st.expander("Extracted context from the uploaded image"):
                st.write(image_text)
            with st.expander("Conversation Scenario"):
                st.write(conversation)

            st.audio(audio_bytes)


def generating_scenario(context, style):
    text_divider = '===='
    prompt = f"""
    Your role is a teacher who prepares English learning materials.
    Make a conversation scenario between a and b based on these context: 
     - Situation: {context}
     - Form of a conversation: {style}
    {text_divider}
    """
    logging.info(prompt)
    model_name = "tiiuae/falcon-7b-instruct"
    # Needs HF pro membership to use
    # model_name = "meta-llama/Llama-2-7b-chat-hf"

    API_URL = f'{API_BASE_URL}{model_name}'
    payload = {
        "inputs": prompt
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    print("Generated text output:", response.json())
    try:
        if len(response.json()) and len(response.json()[0]["generated_text"].split(text_divider)):
            generated_text = response.json()[0]["generated_text"].split(text_divider)[1]
            logging.info("Generated text:", generated_text)
            return generated_text.replace('"', '')
    except KeyError:
        # Error: It should not reach this line in normal cases.
        return context


def img2txt(path):
    # pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    # text = pipe(path)[0]["generated_text"]
    # logging.info("Extracted text:", text)
    # return text
    model_name = "Salesforce/blip-image-captioning-large"
    API_URL = f'{API_BASE_URL}{model_name}'

    with open(path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    if len(response.json()):
        generated_text = response.json()[0]["generated_text"]
        logging.info("Generated text:", generated_text)
        return generated_text.replace('"', '')


def text2speech(input_text):
    model_name = "espnet/kan-bayashi_ljspeech_vits"
    API_URL = f'{API_BASE_URL}{model_name}'
    payload = {
        "inputs": input_text
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    # Create the output directory if it doesn't exist
    dir_path = 'output'
    mkdir(dir_path)

    full_path = os.path.join(dir_path, 'audio.flac')
    with open(full_path, 'wb') as file:
        file.write(response.content)

    return response.content


def text2image(input_text):
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    API_URL = f'{API_BASE_URL}{model_name}'
    payload = {
        "inputs": input_text
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    # Create the output directory if it doesn't exist
    dir_path = 'output'
    mkdir(dir_path)

    full_path = os.path.join(dir_path, 'image.png')
    with open(full_path, 'wb') as file:
        file.write(response.content)

    return response.content


def mkdir(dir_path):
    # Create a directory if it doesn't exist
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        # Handles existing directory gracefully
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    home()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
