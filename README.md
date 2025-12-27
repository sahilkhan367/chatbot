# 🤖 Sahil Khan – AI Assistant (Multimodal RAG + Vision)

An AI-powered personal assistant that answers questions **as Sahil Khan** using a
**Retrieval-Augmented Generation (RAG)** pipeline and also understands images using
**YOLOv8 object detection**.

This project combines **LLMs, Vector Databases, Computer Vision, and Streamlit UI**
to create a smart, multimodal assistant.

---

## 🚀 Features

- 🧠 **Personal RAG Chatbot**
  - Answers questions using Sahil Khan’s personal knowledge
  - Uses PDF-based knowledge with semantic search
  - Responds in first person (“I”, “my”, “me”)

- 📄 **PDF Knowledge Base**
  - Personal data stored in a PDF
  - Chunked, embedded, and indexed using FAISS

- 🖼️ **Image Understanding**
  - Upload an image
  - Detect objects using YOLOv8
  - Explain detected objects using an LLM

- 🔀 **Smart Routing**
  - Text questions → RAG pipeline
  - Image-based queries → Direct LLM (no RAG pollution)

- 💬 **Chat Interface**
  - Streamlit chat UI
  - Chat history preserved during session
  - Thinking indicator for better UX

---

## 🧩 Architecture Overview



User
├── Text Query ──► RAG (FAISS + PDF) ──► LLM ──► Answer
└── Image Upload ─► YOLOv8 ─► Objects ─► LLM ─► Explanation




---

## 🛠️ Tech Stack

- **LLM**: Ollama (Qwen2.5:0.5B)
- **Embeddings**: Ollama Embeddings
- **Vector DB**: FAISS
- **RAG Framework**: LangChain
- **Computer Vision**: YOLOv8 (Ultralytics)
- **Backend**: Python
- **UI**: Streamlit
- **Image Processing**: OpenCV, NumPy

---

## 📁 Project Structure

LLM/
├── app.py # Streamlit UI
├── chatbot.py # RAG pipeline (LLM + FAISS + Prompt)
├── object_detection.py # YOLOv8 object detection
├── rag_backend.py # Vector DB creation script
├── data/
│ └── sahilkhan.pdf # Personal knowledge PDF
├── vectorstore/ # FAISS index
├── env/ # Python virtual environment
└── README.md




└── README.md


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
1️⃣ git clone <your-repo-url>
cd LLM

2️⃣ Create virtual environment
python -m venv env
env\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Install & run Ollama

Download: https://ollama.com

Pull model:

ollama pull qwen2.5:0.5b

5️⃣ Create Vector Database
python rag_backend.py

6️⃣ Run the app
streamlit run app.py

🧪 Example Use Cases
Text Query
Who are you?


➡️

My name is Sahil Khan, I am an IoT Engineer.

Image Upload

Upload a photo

YOLO detects objects (e.g., person, cell phone)

Assistant explains their real-world usage

🧠 Design Decisions (Important)

RAG is only used for personal knowledge

Vision outputs are never sent to RAG

Prevents hallucinations and context pollution

Matches real-world multimodal AI architecture

📌 Limitations

Image explanations are based on object labels (no raw image vision model)

Chat history is session-based (not persisted)

UI follows Streamlit layout constraints

🔮 Future Improvements

🖼️ Vision-language models (Qwen-VL / LLaVA)

🧠 OCR + object detection

💾 Persistent chat memory

🎯 Bounding box visualization

🌐 Web or mobile frontend