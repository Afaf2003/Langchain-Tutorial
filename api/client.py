import requests
import streamlit as st

# Functions to call FastAPI endpoints
def get_openai_response(input_text):
    try:
        response = requests.post(
            "http://localhost:8000/essay/invoke",
            json={'input': {'topic': input_text}}
        )
        response.raise_for_status()
        return response.json().get('output', 'No output found.')
    except requests.exceptions.RequestException as e:
        return f"HTTP error: {e}"
    except ValueError:
        return f"Invalid JSON: {response.text}"

def get_ollama_response(input_text):
    try:
        response = requests.post(
            "http://localhost:8000/poem/invoke",
            json={'input': {'topic': input_text}}
        )
        response.raise_for_status()
        return response.json().get('output', 'No output found.')
    except requests.exceptions.RequestException as e:
        return f"HTTP error: {e}"
    except ValueError:
        return f"Invalid JSON: {response.text}"

# Streamlit UI
st.title('LangChain Demo: Essay and Poem Generator')

input_text = st.text_input("Write an essay on:")
input_text1 = st.text_input("Write a poem on:")

if input_text:
    st.subheader("Essay Output")
    st.write(get_openai_response(input_text))

if input_text1:
    st.subheader("Poem Output")
    st.write(get_ollama_response(input_text1))
