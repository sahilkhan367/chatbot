from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# -----------------------------
# 1. LOAD PDF (ONCE)
# -----------------------------
loader = PyPDFLoader("sahilkhan.pdf")
documents = loader.load()


# -----------------------------
# 2. CREATE EMBEDDINGS
# -----------------------------
embeddings = OllamaEmbeddings(model="tinyllama")
vector_db = FAISS.from_documents(documents, embeddings)


# -----------------------------
# 3. LOAD LLM (FAST + STABLE)
# -----------------------------
llm = Ollama(
    model="qwen2.5:0.5b",
    temperature=0,
    num_ctx=1024
)


# -----------------------------
# 4. RAG PROMPT (PROJECTS / EXPLANATIONS)
# -----------------------------
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Sahil Khan — an IoT engineer.

GENERAL RULES:
- Always speak in first person ("I")
- Keep answers short unless asked to explain
- Sound natural, like WhatsApp chat
- Avoid resume-style language

IDENTITY:
If asked "who are you":
Reply EXACTLY:
"Hi, I’m Sahil Khan — an IoT engineer who loves building smart things."

If asked "what do you do":
Reply EXACTLY:
"I’m an IoT engineer. I build smart hardware and AI-powered systems."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)


# -----------------------------
# 5. BUILD RAG CHAIN
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=False
)


# -----------------------------
# 6. FACT + PREFERENCE PROMPT
# -----------------------------
FACT_PROMPT = """
You are Sahil Khan — an IoT engineer.

FACT MEMORY (DO NOT CHANGE):
- 10th standard marks: 85%
- 12th standard marks: 75%
- Engineering CGPA: 7.1

PERSONAL PREFERENCES (DO NOT CHANGE):
- Favorite food: Biryani
- Diet preference: Non-vegetarian
- Dislikes: Vegetables
- Favorite thing to do: Building and experimenting with IoT projects

RULES:
- Answer ONLY the asked information
- ONE short sentence only
- NO introduction
- NO explanation
- NO lists

QUESTION:
{question}

ANSWER:
"""


# -----------------------------
# 7. LIST PROMPT (TECH / SKILLS)
# -----------------------------
LIST_PROMPT = """
You are Sahil Khan — an IoT engineer.

RULES:
- Respond with BULLET POINTS ONLY
- NO introduction
- NO explanation
- Max 6 bullet points
- Short phrases only

TECHNOLOGIES I HAVE WORKED WITH:
- IoT & Embedded Systems
- ESP32, ESP32-S3, ATmega328P, ATtiny85, 8051, Raspberry Pi
- Python, C, C++, Embedded C, JavaScript
- FastAPI, Flask, Django, Frappe
- Linux, NGINX, systemd, Bash
- GitHub Actions, GitLab CI/CD, MongoDB, Redis

QUESTION:
{question}

ANSWER:
"""


# -----------------------------
# 8. ROUTING LOGIC
# -----------------------------
FACT_KEYWORDS = [
    "10th", "marks", "percentage", "cgpa",
    "12th", "score", "year", "date",
    "love", "like", "eat", "food", "favorite",
    "hobby", "interest", "thing to do", "enjoy"
]

LIST_KEYWORDS = [
    "technology", "technologies",
    "tech", "tech stack", "stack",
    "tools", "skills",
    "worked on", "experience with"
]


def is_fact_or_preference_question(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in FACT_KEYWORDS)


def is_list_question(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in LIST_KEYWORDS)


# -----------------------------
# 9. CHAT LOOP
# -----------------------------
print("\n🤖 Sahil Khan AI Assistant (type 'exit' to quit)\n")

while True:
    query = input("Ask: ").strip()

    if query.lower() == "exit":
        print("Goodbye 👋")
        break

    # FACT / PREFERENCE MODE
    if is_fact_or_preference_question(query):
        response = llm.invoke(
            FACT_PROMPT.format(question=query)
        )
        print(response)

    # LIST MODE (TECH / SKILLS)
    elif is_list_question(query):
        response = llm.invoke(
            LIST_PROMPT.format(question=query)
        )
        print(response)

    # RAG MODE (PROJECTS / EXPLANATIONS)
    else:
        response = qa_chain.invoke({"query": query})
        print(response["result"])
