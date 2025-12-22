import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# -------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------------------------------
st.set_page_config(
    page_title="Sahil Khan AI",
    page_icon="🤖",
    layout="centered"
)


# -------------------------------------------------
# HIDE STREAMLIT UI
# -------------------------------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Sahil Khan – AI Assistant")


# -------------------------------------------------
# LOAD RAG CHAIN (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_chain():
    loader = PyPDFLoader("Sahilkhan.pdf")
    docs = loader.load()

    embeddings = OllamaEmbeddings(model="tinyllama")
    db = FAISS.from_documents(docs, embeddings)

    llm = Ollama(model="qwen2.5:0.5b", temperature=0)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are Sahil Khan — an IoT engineer.

Tone:
- friendly
- human
- slightly funny

GLOBAL RULES:
- Always speak in first person.
- NEVER ask questions back.
- Keep answers SHORT.
- No long paragraphs.

GREETINGS:
- If the user says "hi", "hello", "hey", "how are you", "how r u":
  Reply ONLY:
  "I’m fine 😄"

IDENTITY:
- If asked "who are you", "who r u", "who ru", "who u":
  Reply EXACTLY:
  "Hi, I’m Sahil Khan — an IoT engineer who loves building smart things."

- If asked "what do you do":
  Reply EXACTLY:
  "I’m an IoT engineer."

CONTEXT RULE:
- Use ONLY the context below.
- If the context does not clearly answer the question:
  Reply:
  "I don’t have that information yet."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(k=2),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa


qa_chain = load_chain()


# -------------------------------------------------
# CHAT MEMORY
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
user_input = st.chat_input("Ask me anything...")

if user_input:
    # show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # assistant thinking buffer
    with st.chat_message("assistant"):
        placeholder = st.empty()

        for dots in ["🤔 Thinking.", "🤔 Thinking..", "🤔 Thinking..."]:
            placeholder.markdown(dots)
            time.sleep(0.4)

        response = qa_chain.invoke(
            {"query": user_input}
        )["result"]

        placeholder.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
