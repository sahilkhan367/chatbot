from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os


load_dotenv()



Data_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()   
    return documents

documents = load_pdf_files(data=Data_PATH)
# print("Lenght of pdf documents:", len(documents))

def creat_chunks(extrated_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10
    )
    text_chunks = text_splitter.split_documents(extrated_data)
    return text_chunks

text_chunks = creat_chunks(extrated_data=documents)
print("Lenght of text chunks:", len(text_chunks))



def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


embedding_model=get_embedding_model()
DB_FAISS_PATH="vectorstore/db_faiss" 
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)