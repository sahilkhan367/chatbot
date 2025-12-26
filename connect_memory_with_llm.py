from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ---------------------------
# Embeddings
# ---------------------------
embedding_model = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# Load FAISS
# ---------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# ---------------------------
# Load LLM (TinyLlama)
# ---------------------------
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        temperature=0.5,
        do_sample=True,
    )
    return HuggingFacePipeline(pipeline=pipe)

# ---------------------------
# Prompt
# ---------------------------
CUSTOM_PROMPT_TEMPLATE = """
Use the information in the context to answer the question.
If you don't know the answer, say you don't know.
Do not make up answers.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

# ---------------------------
# RetrievalQA
# ---------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

# ---------------------------
# Run
# ---------------------------
user_query = input("Enter your query: ")
response = qa_chain.invoke({"query": user_query})

print("\nAnswer:\n", response["result"])
