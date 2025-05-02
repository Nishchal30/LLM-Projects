import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF and chat history")
st.write("Upload PDF file and chat with the content")

api_key = st.text_input("Enter the GROQ API key", type="password")

if api_key:
    llm = ChatGroq(api_key = api_key, model_name = "llama3-70b-8192")

    session_id = st.text_input("Enter Session ID", value="default")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            # create a temp file in local to store that pdf file

            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            docs = PyPDFLoader(temp_pdf).load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size= 5000, chunk_overlap = 200)
        final_docs = text_splitter.split_documents(documents=documents)
        vectors = Chroma.from_documents(final_docs, embeddings)
        retriever = vectors.as_retriever()

        context_system_prompt = (
            """
            Given a chat history and the latest user question which might reference context in the chat history, 
            formulate a standalone question which can be understood without the chat history. Do not answer the question, 
            just reformulate it if needed and otherwise return it as it is.
            """
        )

        context_promot = ChatPromptTemplate.from_messages([
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])


        history_aware_retriever = create_history_aware_retriever(llm, retriever, context_promot)

        system_prompt = (
            """
            You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
            \n\n {context}
            """
        )            


        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        qna_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)

        def get_session_history(session_id:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()

            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history, 
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config= {
                    "configurable": {"session_id" : session_id}
                },
            )

            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat history:", session_history.messages)
    
    else:
        st.warning("Please enter the GROQ api key")
