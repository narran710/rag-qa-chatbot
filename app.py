import streamlit as st
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
import os
from dotenv import load_dotenv
import io

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="RAG QA Chatbot", layout="wide")

# 🔥 HIDE DEFAULT FILE UPLOADER PREVIEW (filename + ❌)
st.markdown("""
<style>
section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] + div {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 RAG QA Chatbot")

# -----------------------------
# API KEY
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("❌ GROQ API Key not found! Add it in .env file")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# SESSION STATE
# -----------------------------
defaults = {
    "history": [],
    "view": "main",
    "last_chat": None,
    "file_uploaded": False,
    "chunks": None,
    "index": None,
    "model": None,
    "bm25": None,
    "uploaded_filename": None,
    "file_bytes": None,
    "selected_chats": set(),
    "confirm_delete_selected": False,
    "confirm_delete_all": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# GROQ FUNCTION
# -----------------------------
def ask_groq(context, question, history):
    messages = [
        {"role": "system", "content":
         "Answer ONLY from context. Include chapters, functions, formulas clearly."}
    ]

    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{question}"
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content

# -----------------------------
# FILE READER
# -----------------------------
def read_file(file):
    if file.type == "application/pdf":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return "".join([p.get_text() for p in pdf])
    return file.read().decode("utf-8")

# -----------------------------
# CHUNKING
# -----------------------------
def split_text(text, chunk_size=800, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# -----------------------------
# RETRIEVERS
# -----------------------------
@st.cache_resource(show_spinner=False)
def create_retrievers(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    bm25 = BM25Okapi([c.split() for c in chunks])
    return index, model, bm25

# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve(query, index, model, bm25, chunks, k=8):
    q_embed = model.encode([query])
    _, faiss_idx = index.search(np.array(q_embed), k)

    scores = bm25.get_scores(query.split())
    bm25_idx = np.argsort(scores)[-k:]

    combined = list(set(faiss_idx[0]) | set(bm25_idx))
    return [chunks[i] for i in combined][:5]

# -----------------------------
# CHAT HISTORY VIEW
# -----------------------------
if st.session_state.view == "history":

    st.subheader("📜 Chat History")

    search = st.text_input("🔍 Search history")

    filtered = [
        (i, q, a) for i, (q, a) in enumerate(st.session_state.history)
        if search.lower() in q.lower()
    ]

    if filtered:

        for i, q, a in filtered:
            col1, col2 = st.columns([0.1, 0.9])

            with col1:
                checked = st.checkbox("", key=f"chk_{i}")
                if checked:
                    st.session_state.selected_chats.add(i)
                else:
                    st.session_state.selected_chats.discard(i)

            with col2:
                with st.expander(q):
                    st.write(a)

        st.markdown("---")

        # DELETE SELECTED
        if st.button("🗑 Delete Selected"):
            if st.session_state.selected_chats:
                st.session_state.confirm_delete_selected = True
            else:
                st.warning("No chats selected")

        if st.session_state.confirm_delete_selected:
            st.warning("⚠️ Confirm delete selected chats?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Yes"):
                    st.session_state.history = [
                        item for idx, item in enumerate(st.session_state.history)
                        if idx not in st.session_state.selected_chats
                    ]
                    st.session_state.selected_chats.clear()
                    st.session_state.confirm_delete_selected = False
                    st.rerun()

            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_delete_selected = False
                    st.rerun()

        # DELETE ALL
        if st.button("🗑 Delete All Chats"):
            if st.session_state.history:
                st.session_state.confirm_delete_all = True
            else:
                st.warning("No chats to delete")

        if st.session_state.confirm_delete_all:
            st.warning("⚠️ Confirm delete ALL chats?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔥 Yes Delete All"):
                    st.session_state.history = []
                    st.session_state.selected_chats.clear()
                    st.session_state.confirm_delete_all = False
                    st.rerun()

            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_delete_all = False
                    st.rerun()

    else:
        st.info("No chat history found")

    if st.button("⬅ Back"):
        st.session_state.view = "main"
        st.session_state.last_chat = None
        st.rerun()

# -----------------------------
# MAIN VIEW
# -----------------------------
else:

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"]
    )

    # STORE FILE IN MEMORY
    if uploaded_file is not None:
        st.session_state.file_bytes = uploaded_file.read()
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.file_uploaded = False

    # PROCESS FILE
    if st.session_state.file_bytes and not st.session_state.file_uploaded:

        file_obj = io.BytesIO(st.session_state.file_bytes)

        class FileWrapper:
            def __init__(self, file, name):
                self.file = file
                self.name = name
                self.type = "application/pdf" if name.endswith(".pdf") else "text/plain"

            def read(self):
                return self.file.read()

        wrapped = FileWrapper(file_obj, st.session_state.uploaded_filename)

        text = read_file(wrapped)
        chunks = split_text(text)

        with st.spinner("🔍 Analyzing document..."):
            index, model, bm25 = create_retrievers(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.model = model
        st.session_state.bm25 = bm25
        st.session_state.file_uploaded = True

    # BEFORE UPLOAD
    if not st.session_state.file_uploaded:
        st.info("📄 Upload document to start")

    else:
        st.success("✅ Document ready")
        st.info(f"📄 {st.session_state.uploaded_filename}")

        if st.button("📜 Chat History"):
            st.session_state.view = "history"
            st.rerun()

        if st.button("🔄 Upload New Document"):
            for key in ["chunks", "index", "model", "bm25", "file_bytes"]:
                st.session_state[key] = None

            st.session_state.file_uploaded = False
            st.session_state.history = []
            st.session_state.last_chat = None
            st.session_state.uploaded_filename = None
            st.rerun()

        query = st.chat_input("Ask question...")

        if query:
            with st.spinner("Thinking..."):
                retrieved = retrieve(
                    query,
                    st.session_state.index,
                    st.session_state.model,
                    st.session_state.bm25,
                    st.session_state.chunks
                )

                context = "\n\n".join(retrieved)

                answer = ask_groq(context, query, st.session_state.history)

                st.session_state.history.append((query, answer))
                st.session_state.last_chat = (query, answer)

        # SHOW ONLY LAST CHAT
        if st.session_state.last_chat:
            q, a = st.session_state.last_chat
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(a)
