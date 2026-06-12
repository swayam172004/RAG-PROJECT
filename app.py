import streamlit as st
from RAG_BACKEND import process_pdf, ask_question

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #6366f1,
        #8b5cf6,
        #ec4899
    );
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.4);
}

.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    backdrop-filter: blur(10px);
}

.footer {
    text-align: center;
    padding: 20px;
    color: gray;
}

.stChatMessage {
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.markdown("# 🤖 AI PDF Assistant")

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    process_button = st.button(
        "🚀 Process Document",
        use_container_width=True
    )

    st.markdown("---")

    st.success("Groq Connected")
    st.info("FAISS Vector Store")
    st.warning("RAG Pipeline Active")

# =====================================
# HERO SECTION
# =====================================

st.markdown("""
<div class='hero'>
<h1>📚 RAG Powered PDF Chatbot</h1>
<p>Upload any PDF and chat with your documents using AI</p>
</div>
""", unsafe_allow_html=True)


# =====================================
# PDF PROCESSING
# =====================================

if uploaded_file and process_button:

    with st.spinner("Creating embeddings and indexing document..."):

        process_pdf(uploaded_file)

    st.success(
        "✅ PDF Processed Successfully!"
    )

# =====================================
# CHAT HISTORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# CHAT INPUT
# =====================================

prompt = st.chat_input(
    "Ask anything about your uploaded PDF..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = ask_question(prompt)

            st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

# =====================================
# FOOTER
# =====================================

st.markdown("""
<div class='footer'>
Built by Swayam Sikarwar
</div>
""", unsafe_allow_html=True)
