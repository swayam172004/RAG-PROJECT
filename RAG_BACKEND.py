import numpy as np
import faiss
import tempfile

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------

MODEL = "llama-3.1-8b-instant"

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# Global Variables
# -----------------------------

pdf_index = None
chunks = []

# -----------------------------
# Process PDF
# -----------------------------

def process_pdf(uploaded_file):
    global pdf_index
    global chunks

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    pdf = PdfReader(pdf_path)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    chunk_size = 300

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]

    pdf_index = faiss.IndexFlatL2(dimension)

    pdf_index.add(
        np.array(
            embeddings,
            dtype="float32"
        )
    )

# -----------------------------
# Ask Question
# -----------------------------

def ask_question(question):

    global pdf_index
    global chunks

    if pdf_index is None:
        return "Please upload and process a PDF first."

    query_embedding = embedding_model.encode(
        [question]
    )

    distances, indices = pdf_index.search(
        np.array(
            query_embedding,
            dtype="float32"
        ),
        3
    )

    retrieved_chunks = [
        chunks[i]
        for i in indices[0]
    ]

    context = "\n".join(
        retrieved_chunks
    )

    messages = [
        {
            "role": "system",
            "content": "Answer only from the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    return response.choices[0].message.content
