import streamlit as st
from RAG_BACKEND import process_pdf, ask_question

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🤖 AI PDF Assistant")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    process_button = st.button(
        "Process Document"
    )

# Main UI
st.title("📚 RAG Powered PDF Chatbot")

st.markdown("""
Ask questions about your uploaded document.

Powered by:
- Groq
- FAISS
- Sentence Transformers
- Streamlit
""")

if uploaded_file and process_button:

    with st.spinner("Creating embeddings..."):
        process_pdf(uploaded_file)

    st.success("Document indexed successfully!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input(
    "Ask anything about your PDF..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):

        response = ask_question(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )
