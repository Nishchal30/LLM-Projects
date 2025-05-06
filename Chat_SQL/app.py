import streamlit as st
from pathlib import Path
import sqlite3
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq

st.set_page_config(page_title="Welcome to Chat with SQL DB app!", page_icon="Bird")
st.title("Welcome to Chat with SQL DB app!")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

# Left side of page for config

radio_options = ['Use SQLite 3 DB - student.db', 'Connect to your SQL DB']
selected_option = st.sidebar.radio(label="Select the DB from above options", options=radio_options)

if radio_options.index(selected_option) == 1:
    db_uri = MYSQL

    mysql_host = st.sidebar.text_input("Give the SQL hostname")
    mysql_user = st.sidebar.text_input("Give the SQL user name")
    mysql_password = st.sidebar.text_input("Give the SQL password", type="password")
    mysql_db = st.sidebar.text_input("Give the SQL DB name")
else:
    db_uri = LOCALDB

groq_api_key = st.sidebar.text_input("Give the GROQ API key", type="password")

if not db_uri:
    st.info("Please enter the Database information and select the DB")

if not groq_api_key:
    st.info("Please enter the GROQ API key")

## LLM Model

llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, 
                 mysql_host=None, 
                 mysql_user=None, 
                 mysql_password=None, 
                 mysql_db=None):

    if db_uri == LOCALDB:
        db_file_path = (Path(__file__).parent/"student.db").absolute()
        print(db_file_path)

        creator = lambda: sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_password and mysql_user and mysql_db):
            st.error("Please provide all the MYSQL credentials to connect")
            st.stop()

        else:
            return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))


if db_uri==MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state['messages'] = [
        {"role": "assistant",
         "content": "Hello, I am a chatbot which can retrieve data from DB. How can I help you?"}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message['content'])

user_query = st.chat_input(placeholder="Ask anything here..")

if user_query:
    st.session_state.messages.append({"role":"user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message('assistant'):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

        response = agent.run(user_query, callbacks=[st_callback])
        st.session_state.messages.append({'role':'assistant', 'content': response})

        st.write(response)