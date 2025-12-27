from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load PDF
loader = PyPDFLoader(
    r"C:\Users\user\Desktop\LLM\data\sahilkhan.pdf"
)
documents = loader.load()

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OllamaEmbeddings(model="qwen2.5:0.5b")

# 4. Vector DB
vector_db = FAISS.from_documents(docs, embeddings)

# Save DB
vector_db.save_local("sahil_vector_db")
