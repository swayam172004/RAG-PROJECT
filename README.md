📄 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and ask questions about their content using Large Language Models (LLMs). The application combines semantic search with AI-powered responses to provide accurate answers based on uploaded documents.

🚀 Live Demo

Application: https://rag-project-kej5exwq4zeyst8gs98iaf.streamlit.app/

📌 Overview

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline:

1. Upload PDF documents
2. Extract and process text
3. Generate vector embeddings
4. Store embeddings in a FAISS vector database
5. Retrieve relevant document chunks using semantic search
6. Generate context-aware answers using Groq LLM

The system reduces hallucinations by grounding responses in the uploaded document content.

---

✨ Features

- 📂 PDF Upload Support
- 📖 Automatic Text Extraction
- 🧩 Text Chunking for Efficient Retrieval
- 🔍 Semantic Search using Vector Embeddings
- ⚡ Fast Similarity Search with FAISS
- 🤖 AI-Powered Question Answering using Groq
- 🎯 Context-Aware Responses
- 🌐 Interactive Streamlit Interface

---

🛠️ Tech Stack

Frontend

- Streamlit

Backend

- Python

AI & Machine Learning

- Groq LLM
- Sentence Transformers
- Retrieval-Augmented Generation (RAG)

Vector Database

- FAISS (Facebook AI Similarity Search)

Document Processing

- PyPDF

---

📂 Project Structure

RAG-PROJECT/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── .env                   # API keys (not uploaded to GitHub)
├── README.md              # Project documentation
│
└── data/
    └── uploaded_pdfs/     # Uploaded PDF files

---

⚙️ Installation

1. Clone the Repository

git clone https://github.com/swayam172004/RAG-PROJECT.git

cd RAG-PROJECT

2. Create Virtual Environment

python -m venv venv

Activate Environment

Windows

venv\Scripts\activate

Linux/Mac

source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Configure Environment Variables

Create a ".env" file in the project root:

GROQ_API_KEY=your_groq_api_key

5. Run the Application

streamlit run app.py

---

🔄 How It Works

Step 1: Upload PDF

The user uploads a PDF document through the Streamlit interface.

Step 2: Text Extraction

Text is extracted from the uploaded PDF using PyPDF.

Step 3: Embedding Generation

The extracted text is converted into vector embeddings using Sentence Transformers.

Step 4: Vector Storage

Embeddings are stored in a FAISS vector database for efficient similarity search.

Step 5: Retrieval

When a question is asked, the most relevant document chunks are retrieved from FAISS.

Step 6: Response Generation

The retrieved context is sent to the Groq LLM, which generates an accurate and context-aware response.

---

🎯 Use Cases

- Research Paper Analysis
- Academic Study Assistant
- Document Q&A System
- Company Policy Search
- Legal Document Exploration
- Technical Documentation Assistant
- Knowledge Base Chatbot

---

📸 Application Workflow

PDF Upload
     │
     ▼
Text Extraction
     │
     ▼
Text Chunking
     │
     ▼
Embeddings Generation
     │
     ▼
FAISS Vector Store
     │
     ▼
Semantic Retrieval
     │
     ▼
Groq LLM
     │
     ▼
Generated Answer

---

🔒 Security Notes

- Never commit API keys to GitHub.
- Store secrets using Streamlit Secrets or environment variables.
- Add ".env" to ".gitignore".

Example:

.env

---

🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

👨‍💻 Author

Swayam Sikarwar

- GitHub: https://github.com/swayam172004

---

⭐ Support

If you found this project useful, consider giving the repository a star. It helps others discover the project and supports future development.