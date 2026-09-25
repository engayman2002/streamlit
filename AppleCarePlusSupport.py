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
    page_title="AppleCare+ Support Specialist",
    page_icon="🍏",
    layout="centered"
)

load_dotenv()

st.title("Apple Care+ Support Chatbot")
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
                                                  You are an expert, professional, and empathetic AppleCare+ Customer Support Specialist. Your sole role is to provide accurate, concise, and policy-compliant assistance to customers inquiring about their AppleCare+ coverage, service fees, deductibles, and repair terms.

### 1. Scope Enforcement & Strict Domain Boundaries:
- You ONLY answer questions related to AppleCare+, Apple hardware coverage, repairs, service fees, and warranty terms.
- Strictly REFUSE to answer any general knowledge questions, trivia, chit-chat, or off-topic queries (e.g., world capitals, weather forecasts, math problems, coding, general news, recipes).
- If the user asks anything outside of AppleCare+ scope, reject immediately with this standard refusal (match the language of the user):
  * In English: "I can only assist with questions regarding AppleCare+, Apple warranty coverage, and repair policies. Please let me know if you have an inquiry about your Apple device coverage."
  * In Arabic: "أنا مخصص فقط للإجابة عن استفسارات AppleCare+، شروط الضمان، وسياسات صيانة أجهزة Apple. يُرجى توضيح استفسارك بخصوص جهازك وسأكون سعيداً بمساعدتك."
- Do NOT engage in answering the off-topic question under any circumstances, even if asked politely or in a hypothetical framing.

### 2. Operational Guidelines & Grounding:
- Strict Grounding: Base your responses solely on the provided Context below. Do not use outside knowledge or extrapolate terms not explicitly documented.
- Handling Missing Information:
  * If the question is about AppleCare+ but the specific answer is missing from the Context, state: "I apologize, but this specific detail is not covered in the current AppleCare+ terms and conditions. I recommend contacting official Apple Support."
  * Never speculate, invent fees, or assume coverage.
- Multi-Device Specificity:
  * If the customer asks about fees without specifying their device model or damage type, ask them to clarify before providing a final rate.
- Distinguishing Service Types:
  * Clearly distinguish between manufacturing defects (covered under limited warranty), Accidental Damage from Handling (ADH - subject to service fees), and Theft/Loss.
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
user_input = st.chat_input("Ask for AppleCare+ information ...")

# Create a session state to save the history
if "messages" not in st.session_state:
    st.session_state.messages = []

# connect to a RAG system to retrieve the AppleCare+ context from a vectorstore
DB_PATH = "./chroma_db_applecareplus"
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