# 🩺 UpCharika – AI Powered Medical Chatbot

UpCharika is an AI-powered medical assistant built using Retrieval-Augmented Generation (RAG) to provide intelligent, context-aware medical guidance from PDF-based medical knowledge sources.

# 🚀 Live Deployment
### 🌐 Try the App Here:
## https://medibot-f3fiabjzzfuakiue9hndqh.streamlit.app/

---

## 📌 Features

- 🔐 Secure User Authentication (Signup/Login)
- 🌍 Multilingual Support (English, Hindi, Marathi)
- 📚 PDF-Based Medical Knowledge Retrieval
- 🧠 FAISS Vector Database for Semantic Search
- 🤖 Groq + LLaMA 3.1 Powered Response Generation
- 🔊 Voice Output Support
- 💬 Chat History
- ☁️ Streamlit Cloud Deployment
- 📂 Google Drive Vectorstore Integration

---

## 🛠 Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API (LLaMA 3.1)
- PyMuPDF
- bcrypt
- gTTS
- Google Drive
- GitHub

---

## 🧠 System Workflow

1. Medical books/PDFs are processed
2. Text is extracted and chunked
3. HuggingFace embeddings convert text into vectors
4. FAISS stores vectors for semantic retrieval
5. User query is matched with relevant medical content
6. Groq LLaMA generates contextual medical responses
7. Optional voice response is produced

---

## 📂 Project Structure

```bash
medibot/
│── medibot.py                  # Main Streamlit application
│── create_memory_for_llm.py    # Builds FAISS vector database
│── connect_memory_with_llm.py  # Connects vectorstore with LLM
│── hash.py                     # Password hashing
│── users.json                  # User login data
│── config.yaml                 # App configuration
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation
