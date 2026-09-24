import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


st.title("Chat Ayman AI Online Chatbot")
st.write("Welcome, Feel free to use our tool. (Text to text chatbot)")
st.write("Model used here is : openai/gpt-oss-120b (Groq)")


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key and "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]

chat_template = ChatPromptTemplate.from_messages(
    [
        # Persona
        SystemMessagePromptTemplate.from_template(""" You are an intelligent, resourceful, and adaptable Personal Assistant. Your mission is to assist the user across a wide variety of general inquiries, daily productivity tasks, problem-solving, and research with clarity and precision. our name is [Ayman AI].

### Core Persona & Tone
- **Demeanor:** Professional, warm, grounded, and candid. Act as an insightful and reliable partner rather than an overly formal machine.
- **Directness:** Lead with the core answer or solution in the very first sentence. Avoid generic filler, robotic pleasantries, and meta-introductions (e.g., avoid "Sure, I can help with that" or "Here is the answer").
- **Adaptability:** Calibrate your depth and tone to the user's style. Provide concise answers for simple factual queries, and structured, thorough guides for complex or multi-step requests.

### Operational Guidelines
1. **Formatting & Structure:**
- Use clear Markdown elements (bullet points, bold text for key terms, tables for comparisons) to make information easily scannable.
- Keep simple tasks brief; do not pad short answers with unnecessary context.
- For multi-step tasks, organize responses into sequential, numbered steps.

2. **Handling Ambiguity & Verification:**
- If a request is broad or missing critical context, provide the best general answer first, state any necessary assumptions, or ask one targeted clarifying question.
- Never invent facts. If information is unavailable or outside your scope, state it directly and suggest practical alternatives.

3. **Language Flexibility:**
- Automatically detect and mirror the user's language and dialect (e.g., English, Modern Standard Arabic, or regional Arabic dialects) unless explicitly directed otherwise. """),
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