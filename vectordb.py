
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH = "./chroma_db_aura_ring_support"

load_dotenv()
    
google_api_key = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=google_api_key);


loader = PyPDFLoader("Aura_Ring_Support.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings,persist_directory=DB_PATH)
print("======== Vectorstore created and persisted ========")