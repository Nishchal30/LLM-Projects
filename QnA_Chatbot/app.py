from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI powered helpful assistant. Please repsond to user queries in bit of a funny way"),
    ("user", "Question: {question}")
])

def generate_response(question, api_key, model, temperature, max_tokens):

    llm = ChatGroq(model = model, groq_api_key = api_key)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question': question})
    return answer

st.title("Welcome to open source Q&A chatbot")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ api key:", type="password")
model = st.sidebar.selectbox("Select the open source model to run the Chatbot", ["qwen-qwq-32b", "llama-3.1-8b-instant", "mistral-saba-24b", "llama3-70b-8192", "deepseek-r1-distill-llama-70b"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Go ahead and play with the Chatbot")
user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, api_key, model, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please ask any question to start the Chatbot")