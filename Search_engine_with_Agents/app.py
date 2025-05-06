import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
# from langchain_community.embeddings import HuggingFaceEmbeddings


arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

search = DuckDuckGoSearchRun(name = "Google-Search")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("Welcome to Langchain chat with Search")

st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter your GROQ api key:", type="password")

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state['messages'] = [
        {"role": "assistant",
         "content": "Hello, I am a chatbot who can search on the web. How can I help you?"}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message['content'])

if prompt:= st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192", streaming=True)

    tools = [arxiv, wiki, search]
    search_agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors = True)

    with st.chat_message('assistant'):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

        response = search_agent.run(st.session_state.messages, callbacks=[st_callback])
        st.session_state.messages.append({'role':'assistant', 'content': response})

        st.write(response)