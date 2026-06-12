{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0cc171d0",
   "metadata": {},
   "source": [
    "# RAG "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0bd5d181",
   "metadata": {},
   "source": [
    "## Install Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a4ddad0",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install groq sentence-transformers faiss-cpu pypdf numpy -q"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "65a4c41f",
   "metadata": {},
   "source": [
    "## Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76d5614c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import faiss\n",
    "\n",
    "from groq import Groq\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from pypdf import PdfReader"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41e05e58",
   "metadata": {},
   "source": [
    "## Setup Groq API"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "976cb126",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from groq import Groq\n",
    "\n",
    "client = Groq(\n",
    "    api_key=st.secrets[\"GROQ_API_KEY\"]\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a395a5e2-727c-47cd-bbd7-621dbad3cd85",
   "metadata": {},
   "outputs": [],
   "source": [
    "client = Groq(\n",
    "    api_key=GROQ_API_KEY\n",
    ")\n",
    "\n",
    "MODEL = \"llama-3.1-8b-instant\" "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b6f58417",
   "metadata": {},
   "source": [
    "## Ask LLM Without RAG"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40c63439",
   "metadata": {},
   "outputs": [],
   "source": [
    "messages = [\n",
    "    {\n",
    "        \"role\": \"user\",\n",
    "        \"content\": \"What is inside my private PDF?\"\n",
    "    }\n",
    "]\n",
    "\n",
    "response = client.chat.completions.create(\n",
    "    model=MODEL,\n",
    "    messages=messages\n",
    ")\n",
    "\n",
    "print(response.choices[0].message.content)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "628e0a37",
   "metadata": {},
   "source": [
    "## Load Embedding Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2435236a",
   "metadata": {},
   "outputs": [],
   "source": [
    "embedding_model = SentenceTransformer(\"all-MiniLM-L6-v2\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "34895e4d",
   "metadata": {},
   "source": [
    "## Create Sample Documents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b1b2612",
   "metadata": {},
   "outputs": [],
   "source": [
    "documents = [\n",
    "    \"RAG stands for Retrieval Augmented Generation.\",\n",
    "    \"Embeddings convert text into vectors.\",\n",
    "    \"FAISS is used for similarity search.\",\n",
    "    \"Chunking splits large documents into smaller pieces.\"\n",
    "]\n",
    "\n",
    "print(documents)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "18167ebd",
   "metadata": {},
   "source": [
    "## Generate Embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fab2058a",
   "metadata": {},
   "outputs": [],
   "source": [
    "embeddings = embedding_model.encode(documents)\n",
    "\n",
    "print(embeddings.shape)\n",
    "print(embeddings[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "667a32b6",
   "metadata": {},
   "source": [
    "## Create FAISS Index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0d2b872",
   "metadata": {},
   "outputs": [],
   "source": [
    "dimension = embeddings.shape[1]\n",
    "\n",
    "index = faiss.IndexFlatL2(dimension)\n",
    "\n",
    "index.add(np.array(embeddings, dtype=\"float32\"))\n",
    "\n",
    "print(\"Total vectors stored:\", index.ntotal)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "686f99b3",
   "metadata": {},
   "source": [
    "## Ask a Question"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a86cdfca",
   "metadata": {},
   "outputs": [],
   "source": [
    "query = \"What does RAG stand for?\"\n",
    "\n",
    "query_embedding = embedding_model.encode([query])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04f5197e",
   "metadata": {},
   "source": [
    "## Search Similar Chunks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b08b2018",
   "metadata": {},
   "outputs": [],
   "source": [
    "distances, indices = index.search(\n",
    "    np.array(query_embedding, dtype=\"float32\"),\n",
    "    2\n",
    ")\n",
    "\n",
    "print(indices)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9cea90f3",
   "metadata": {},
   "source": [
    "## Retrieve Matching Documents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09079438",
   "metadata": {},
   "outputs": [],
   "source": [
    "retrieved_docs = [documents[i] for i in indices[0]]\n",
    "\n",
    "print(retrieved_docs)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "78dd31db",
   "metadata": {},
   "source": [
    "## Create Context"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "beca0bfe",
   "metadata": {},
   "outputs": [],
   "source": [
    "context = \"\\n\".join(retrieved_docs)\n",
    "\n",
    "print(context)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bb7d109a",
   "metadata": {},
   "source": [
    "## Send Context to LLM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3caa6eaf",
   "metadata": {},
   "outputs": [],
   "source": [
    "messages = [\n",
    "    {\n",
    "        \"role\": \"system\",\n",
    "        \"content\": \"Answer only using the provided context.\"\n",
    "    },\n",
    "    {\n",
    "        \"role\": \"user\",\n",
    "        \"content\": f\"Context:\\n{context}\\n\\nQuestion: {query}\"\n",
    "    }\n",
    "]\n",
    "\n",
    "response = client.chat.completions.create(\n",
    "    model=MODEL,\n",
    "    messages=messages\n",
    ")\n",
    "\n",
    "print(response.choices[0].message.content)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2366a87d",
   "metadata": {},
   "source": [
    "## Load PDF File"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de896ce8",
   "metadata": {},
   "outputs": [],
   "source": [
    "pdf = PdfReader(\"samplerag.pdf\")\n",
    "\n",
    "text = \"\"\n",
    "\n",
    "for page in pdf.pages:\n",
    "    text += page.extract_text()\n",
    "\n",
    "print(text[:1000])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5fe0362a",
   "metadata": {},
   "source": [
    "## Chunk the PDF Text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ef36810",
   "metadata": {},
   "outputs": [],
   "source": [
    "chunk_size = 300\n",
    "\n",
    "chunks = []\n",
    "\n",
    "for i in range(0, len(text), chunk_size):\n",
    "    chunk = text[i:i + chunk_size]\n",
    "    chunks.append(chunk)\n",
    "\n",
    "print(\"Total Chunks:\", len(chunks))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97cec51c",
   "metadata": {},
   "source": [
    "## Generate Chunk Embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ee6beaf",
   "metadata": {},
   "outputs": [],
   "source": [
    "chunk_embeddings = embedding_model.encode(chunks)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a475788f",
   "metadata": {},
   "source": [
    "## Store Chunks in FAISS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea4ae7bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "dimension = chunk_embeddings.shape[1]\n",
    "\n",
    "pdf_index = faiss.IndexFlatL2(dimension)\n",
    "\n",
    "pdf_index.add(np.array(chunk_embeddings, dtype=\"float32\"))\n",
    "\n",
    "print(pdf_index.ntotal)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7afec416",
   "metadata": {},
   "source": [
    "## Ask Questions on PDF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3912038f",
   "metadata": {},
   "outputs": [],
   "source": [
    "query = \"What is the document about?\"\n",
    "\n",
    "query_embedding = embedding_model.encode([query])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06414847",
   "metadata": {},
   "source": [
    "## Retrieve Relevant PDF Chunks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64b9d93a",
   "metadata": {},
   "outputs": [],
   "source": [
    "distances, indices = pdf_index.search(\n",
    "    np.array(query_embedding, dtype=\"float32\"),\n",
    "    3\n",
    ")\n",
    "\n",
    "retrieved_chunks = [chunks[i] for i in indices[0]]\n",
    "\n",
    "print(retrieved_chunks)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a749bce5",
   "metadata": {},
   "source": [
    "## Generate Final RAG Answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c05cfa5",
   "metadata": {},
   "outputs": [],
   "source": [
    "context = \"\\n\".join(retrieved_chunks)\n",
    "\n",
    "messages = [\n",
    "    {\n",
    "        \"role\": \"system\",\n",
    "        \"content\": \"Answer only from the provided context.\"\n",
    "    },\n",
    "    {\n",
    "        \"role\": \"user\",\n",
    "        \"content\": f\"Context:\\n{context}\\n\\nQuestion: {query}\"\n",
    "    }\n",
    "]\n",
    "\n",
    "response = client.chat.completions.create(\n",
    "    model=MODEL,\n",
    "    messages=messages\n",
    ")\n",
    "\n",
    "print(response.choices[0].message.content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d91a2c6-8589-4380-a567-9c3e53a6c486",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip insta"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
