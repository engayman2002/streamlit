import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import random

st.title("ChatAymanAI")
st.write("Welcome, Feel free to use our tool.")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

chat_template = ChatPromptTemplate.from_messages(
    [
        # Persona
        SystemMessagePromptTemplate.from_template("You are a good teacher, please explain my question"),
        # History
        MessagesPlaceholder(variable_name="chat_history"),
        # Query
        HumanMessagePromptTemplate.from_template("My Question : {Question}")
    ]
)

chat_model = ChatGroq(model="openai/gpt-oss-120b",api_key=groq_api_key,temperature=1)

Parser = StrOutputParser()

mychain = chat_template | chat_model | Parser

# User Input
user_input = st.chat_input("Ask Something ...")

# Create a session state to save the history
if "messages" not in st.session_state:
    st.session_state.messages = []

history = []
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])
    
    if message["role"] == "user":
        history.append(HumanMessage(content=message["content"]))
    elif message["role"] == "ai":
        history.append(AIMessage(content=message["content"]))

if user_input is not None:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role":"user","content":user_input})

    #bot_response = mychain.invoke(user_input)
    #st.chat_message("ai").markdown(bot_response)

    with st.chat_message("ai"):
        streamed_text = st.write_stream(mychain.stream({"chat_history":history,"Question":user_input}))

    st.session_state.messages.append({"role":"ai","content":streamed_text})