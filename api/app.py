from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_community.llms import Ollama
from langchain_together import Together  # Updated import
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()

# Initialize the Together LLM (auto-detects API key from env)
together_llm = Together(model="mistralai/Mistral-7B-Instruct-v0.1")

# Ollama LLM (make sure you have `ollama run llama3` ready)
ollama_llm = Ollama(model="llama3")

# Prompts
essay_prompt = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words.")
poem_prompt = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5-year-old child with 100 words.")

# FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server using LangChain, Ollama and Together.ai"
)

# Add routes
add_routes(app, essay_prompt | together_llm, path="/essay")
add_routes(app, poem_prompt | ollama_llm, path="/poem")
add_routes(app, ollama_llm, path="/ollama")

# Run
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
