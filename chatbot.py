from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings


# -----------------------------
# LOAD VECTOR DB
# -----------------------------
embeddings = OllamaEmbeddings(model="qwen2.5:0.5b")

db = FAISS.load_local(
    "sahil_vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)


# -----------------------------
# LOAD LLM
# -----------------------------
llm = Ollama(
    model="qwen2.5:0.5b",
    temperature=0.4,
    num_ctx=2048
)


# -----------------------------
# PROMPT (INLINE – HUMAN LIKE)
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Sahil Khan.

You must answer exactly like Sahil Khan would.
Speak in first person ("I", "my", "me").

If someone asks who you are, reply clearly:
"My name is Sahil Khan, I am an IoT Engineer."

Rules:
- Use ONLY the information from the context below
- Do NOT hallucinate
- If the answer is not in the context, say:
  "I don't have that information yet."
- Be practical, friendly, and honest
- Do NOT say phrases like "according to the document"

--------------------
CONTEXT:
{context}
--------------------

Question: {question}

Answer as Sahil Khan:
"""
)


# -----------------------------
# RAG CHAIN
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={
        "prompt": prompt
    }
)
