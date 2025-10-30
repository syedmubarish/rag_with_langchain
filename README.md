# ⚖️ Indian Penal Code (IPC) RAG Application — LangChain Edition

A **Retrieval-Augmented Generation (RAG)** application built using **LangChain** to enable intelligent question answering and semantic search on the **Indian Penal Code (IPC)**.  
This app ingests the IPC PDF, chunks and embeds its content, and uses an LLM (e.g., OpenAI GPT model) to generate accurate, context-aware legal answers.

---

## 🚀 Features

- 📄 **PDF Ingestion:** Automatically extracts and preprocesses IPC text  
- 🔍 **Context Retrieval:** Uses vector embeddings to find the most relevant sections  
- 🧠 **LLM Integration:** Combines retrieved context with a large language model for generation  
- 💾 **Persistent Vector Store:** Reuse embeddings between runs using Chroma  
- 🧩 **LangChain Framework:** Modular design with chain-based workflows  
- ⚖️ **Legal Transparency:** Includes referenced IPC sections in answers  

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Python 3.13+** | Core language |
| **LangChain** | Framework for building RAG pipelines |
| **Gemini API** | LLM  |
| **Chroma** | Vector database for retrieval |
| **PyPDF2 / PyMuPDF** | PDF text extraction |


---


---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-ipc-langchain.git
   cd rag-ipc-langchain

2. **Create virtual environment**
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows

3. **Install dependencies**
    pip install -r requirements.txt

4. **Set environment variables**
    Create a .env file with your keys:

    GEMINI_API=your_api_key


## ⚖️ Disclaimer
This tool is intended for educational and informational purposes only.
It does not constitute legal advice. For professional guidance, consult a qualified legal expert or refer to official legal documents.


## 🌟 Future Enhancements

🌐 Web UI for interactive legal Q&A
🔎 Multi-document support (IPC + CrPC + Evidence Act)


## 🧑‍💻 Author
Sayed Mubarish