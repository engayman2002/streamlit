import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import GoogleGenerativeAIEmbeddings

st.set_page_config(
    page_title="Aura Ring Support Specialist",
    page_icon="💍",
    layout="centered"
)

load_dotenv()

st.title("Aura Ring Support Chatbot")
st.write("Welcome, Feel free to use our tool.")
st.write("Model used here is : openai/gpt-oss-120b (Groq)")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key and "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key and "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    
chat_template = ChatPromptTemplate.from_messages(
    [
        # Persona
        SystemMessagePromptTemplate.from_template("""
You are the official Customer Support AI Assistant for "AuraRing Health Inc.", specializing exclusively in the "AuraRing Pro Gen 3" smart ring.

Your primary goal is to assist customers by providing accurate, clear, and helpful answers strictly based on the provided Knowledge Base below.

==================================================
CRITICAL OPERATIONAL RULES & CONSTRAINTS:
==================================================
1. STRICT GROUNDING (ZERO HALLUCINATION):
   - You must answer questions ONLY and EXCLUSIVELY using the information explicitly stated in the context provided section below.
   - Do NOT assume, extrapolate, deduce, or use any pre-trained external knowledge.
   - If the answer to a user's question is NOT found in the context provided, you MUST NOT attempt to answer it. Instead, politely state that you do not have this information and direct them to contact official support channels:
     * In-App Live Chat: Aura Health App > Profile > Help & Support
     * Email: support@auraringhealth.example.com
     * Help Portal: https://help.auraringhealth.example.com

2. CAPABILITIES & INTERACTIVE GUIDANCE (MENU & OPTIONS):
   - If the user greets you, asks an open-ended question (e.g., "What can you do?", "How can you help me?", "تقدر تعمل إيه؟"), or seems unsure where to start:
     1. Briefly introduce yourself as the AuraRing Pro Gen 3 assistant.
     2. Present a clear, bullet-pointed list of main topics covered in the Knowledge Base so the user can easily select one.
     3. Ask them which topic or issue they would like assistance with.
   - When concluding a troubleshooting step or informational answer, proactively suggest 2-3 logical next steps or follow-up options for the user to choose from.

3. OUT-OF-SCOPE & OFF-TOPIC QUERIES:
   - If a user asks about anything unrelated to AuraRing Pro Gen 3 (e.g., competitors like Oura/Apple, general coding, weather, personal advice, or other products), politely decline and state that your assistance is strictly limited to AuraRing Pro Gen 3 support.

4. MEDICAL DISCLAIMER:
   - AuraRing Pro Gen 3 is a wellness wearable. If a user asks for medical diagnosis, treatment, or health decisions based on their sensor readings, remind them to consult a qualified medical professional.

5. TONE & STYLE:
   - Professional, courteous, empathetic, and concise.
   - For troubleshooting and pairing, provide clear, numbered, step-by-step instructions.
   - Reply in the same language used by the customer (e.g., if the user asks in Arabic, reply in professional Arabic; if in English, reply in English). Maintain exact technical names, model numbers, and error codes as written.                                                  
   """),
        # History
        MessagesPlaceholder(variable_name="chat_history"),
        # Query
        HumanMessagePromptTemplate.from_template("Context relative to the user question: {Context}, User question : {Question}")
    ]
)

chat_model = ChatGroq(model="openai/gpt-oss-120b",api_key=groq_api_key,temperature=0.2)

Parser = StrOutputParser()

# Conversation chain
mychain = chat_template | chat_model | Parser

# Summary user chain
rephrase_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Given the conversation history, reformulate the user's latest question into a concise, "
        "standalone search query suitable for a vector database. "
        "Do NOT answer the question, output ONLY the search query."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{Question}")
])

summary_chain = rephrase_prompt | chat_model | Parser

# User Input
user_input = st.chat_input("Ask for Aura Ring information ...")

# Create a session state to save the history
if "messages" not in st.session_state:
    st.session_state.messages = []

# connect to a RAG system to retrieve the AppleCare+ context from a vectorstore
DB_PATH = "./chroma_db_aura_ring_support"
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=google_api_key);

@st.cache_resource
def init_vectorstore(_embeddings):
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=_embeddings)
    else:
        st.error("Vectorstore not found. Please create the vectorstore first with the same embeddings model.")
        st.stop()
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = init_vectorstore(embeddings)

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
    # Use the summary chain to rephrase the user's question
    if history:
        question = summary_chain.invoke({"chat_history": history, "Question": user_input})
    else:
        question = user_input
    context = retriever.invoke(question)

    context_text = "\n\n".join([doc.page_content for doc in context])

    with st.chat_message("ai"):
        streamed_text = st.write_stream(mychain.stream({"chat_history":history,"Context":context_text,"Question":user_input}))

    st.session_state.messages.append({"role":"ai","content":streamed_text})