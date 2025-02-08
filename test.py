from dotenv import load_dotenv
load_dotenv()  # Load all the environment variables

import streamlit as st
import os
import google.generativeai as genai
import groq
from PIL import Image
from langchain.llms import Llama

# Configure API keys
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_api_key = os.getenv("GROQ_API_KEY")
llama_model_name = os.getenv("LLAMA_MODEL_NAME")  # Assuming you have the model name as an environment variable

# Function to load Google Gemini Pro Vision API And get response
def get_gemini_response(input_text, image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, image[0], prompt])
    return response.text

# Function to get response from Groq
def get_groq_response(input_text):
    groq_client = groq.Groq(api_key=groq_api_key)
    response = groq_client.chat.completions.create(messages=[{"role": "user", "content": input_text}])
    return response.choices[0].message["content"]

# Function to get response from Llama
def get_llama_response(input_text):
    llama = Llama(model_name=llama_model_name)
    response = llama(input_text)
    return response

# Function to handle image setup
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize our Streamlit app
st.set_page_config(page_title="Gemini Health App")
st.header("Gemini Health App")

input_text = st.text_input("Input Prompt:", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me the total calories")

input_prompt = """
You are an expert nutritionist. You need to see the food items from the image
and calculate the total calories. Also, provide the details of every food item with calorie intake
in the format below:

1. Item 1 - number of calories
2. Item 2 - number of calories
----
----
"""

# If submit button is clicked
if submit:
    image_data = input_image_setup(uploaded_file)
    gemini_response = get_gemini_response(input_text, [image], input_prompt)
    groq_response = get_groq_response(input_prompt)
    llama_response = get_llama_response(input_text)

    st.subheader("Responses:")
    st.write("Gemini Response:", gemini_response)
    st.write("Groq Response:", groq_response)
    st.write("Llama Response:", llama_response)
