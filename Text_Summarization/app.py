import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.documents import Document 



st.set_page_config(page_title="Welcome to URL summarie app!")
st.title("Welcome to text summarization app!")
st.subheader("Summarize URL")

st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter your GROQ api key:", type="password")
model = st.sidebar.selectbox("Select the open source model to run the app", ["llama3-70b-8192", "gemma2-9b-it", "llama3-8b-8192", "mistral-saba-24b", "qwen-qwq-32b", "deepseek-r1-distill-llama-70b"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

llm = ChatGroq(api_key=groq_api_key, model=model, streaming=True, temperature=temperature, max_tokens=max_tokens)

prompt_template = """
Provide a summary of the following content in 300 words
Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

url = st.text_input("URL", label_visibility="hidden")

if st.button("Summarize the content from the given URL"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide the api key and URL")

    elif not validators.url(url):
        st.error("Please provide the valid URL, it can may be a Youtube video URL or any website URL")

    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(youtube_url=url)
                    st.info("Detected YouTube URL. Attempting to fetch transcript.")
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False)
                
                data = loader.load()
                ## Chain for summarization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt= prompt)
                output_summary = chain.run(data)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Error as: {e}")
            st.warning(f"YoutubeLoader failed, trying fallback. Error: {e}")
            


