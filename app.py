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
st.title("🤖 RAG QA Chatbot")

st.markdown("""
<style>
[data-testid="stFileUploader"] {
    width: 100%;
}

[data-testid="stFileUploaderDropzone"] {
    padding: 1.5rem;
    border-radius: 12px;
}

[data-testid="stFileUploaderDropzone"] section {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------
# API KEY
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("❌ GROQ API Key not found!")
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
         """You are a document QA assistant.
        Answer ONLY using the provided context.
        If answer is not in context, say 'Not found in document'.
        Be precise and structured."""}
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
def split_text(text, chunk_size=300, overlap=100):
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
    embeddings = model.encode(chunks, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine similarity
    index.add(np.array(embeddings))

    bm25 = BM25Okapi([c.split() for c in chunks])
    return index, model, bm25

# -----------------------------
# RETRIEVE WITH COSINE
# -----------------------------
def retrieve(query, index, model, bm25, chunks, k=10):

    # 🔹 Dense retrieval (FAISS)
    q_embed = model.encode([query], normalize_embeddings=True)[0]
    scores, faiss_idx = index.search(np.array([q_embed]), k)

    # 🔹 Sparse retrieval (BM25)
    bm25_scores = bm25.get_scores(query.split())
    bm25_idx = np.argsort(bm25_scores)[-k:]

    # 🔹 Combine both
    combined_idx = list(set(faiss_idx[0]) | set(bm25_idx))

    # 🔹 Re-rank using cosine similarity
    chunk_embeddings = model.encode(
        [chunks[i] for i in combined_idx],
        normalize_embeddings=True
    )

    similarities = []

    for i in range(len(combined_idx)):
        sim = np.dot(q_embed, chunk_embeddings[i])

        # 🔥 amplify signal slightly (safe scaling)
        sim = sim ** 0.5

        similarities.append(sim)

    # 🔥 Sort by similarity
    sorted_pairs = sorted(zip(similarities, combined_idx), reverse=True)

    # 🔥 Return top 5 chunks
    filtered = [(chunks[idx], sim) for sim, idx in sorted_pairs if sim > 0.6]

    return filtered[:5] if filtered else [(chunks[idx], sim) for sim, idx in sorted_pairs[:5]]

# -----------------------------
# CONFIDENCE SCORE
# -----------------------------
def confidence_score(chunks, answer, model):
    chunk_emb = model.encode(chunks, normalize_embeddings=True)
    ans_emb = model.encode([answer], normalize_embeddings=True)[0]

    sims = [np.dot(ans_emb, emb) for emb in chunk_emb]

    # 🔥 Take top 3 average instead of max
    top_k = sorted(sims, reverse=True)[:3]

    return sum(top_k) / len(top_k)

# -----------------------------
# CHAT HISTORY VIEW
# -----------------------------
# -----------------------------
# CHAT HISTORY VIEW
# -----------------------------
if st.session_state.view == "history":

    st.subheader("📜 Chat History")

    search = st.text_input("🔍 Search history")

    filtered = [
        (i, q, a)
        for i, (q, a) in enumerate(st.session_state.history)
        if search.lower() in q.lower()
    ]

    if filtered:

        st.write("### Select chats")

        for i, q, a in filtered:

            col1, col2 = st.columns([0.08, 0.92])

            with col1:
                checked = st.checkbox(
                    "",
                    value=i in st.session_state.selected_chats,
                    key=f"chat_{i}"
                )

                if checked:
                    st.session_state.selected_chats.add(i)
                else:
                    st.session_state.selected_chats.discard(i)

            with col2:
                with st.expander(q):
                    st.write(a)

        st.markdown("---")

        col1, col2 = st.columns(2)

        # DELETE SELECTED
        with col1:
            if st.button("🗑 Delete Selected"):
                if st.session_state.selected_chats:
                    st.session_state.confirm_delete_selected = True
                else:
                    st.warning("No chats selected")

        # DELETE ALL
        with col2:
            if st.button("🗑 Delete All"):
                if st.session_state.history:
                    st.session_state.confirm_delete_all = True
                else:
                    st.warning("No chats to delete")

        # CONFIRM DELETE SELECTED
        if st.session_state.confirm_delete_selected:

            st.warning("⚠️ Delete selected chats?")

            c1, c2 = st.columns(2)

            with c1:
                if st.button("✅ Yes Delete Selected"):

                    st.session_state.history = [
                        item
                        for idx, item in enumerate(st.session_state.history)
                        if idx not in st.session_state.selected_chats
                    ]

                    st.session_state.selected_chats.clear()
                    st.session_state.confirm_delete_selected = False

                    st.success("Selected chats deleted")
                    st.rerun()

            with c2:
                if st.button("❌ Cancel Selected"):
                    st.session_state.confirm_delete_selected = False
                    st.rerun()

        # CONFIRM DELETE ALL
        if st.session_state.confirm_delete_all:

            st.warning("⚠️ Delete ALL chats?")

            c1, c2 = st.columns(2)

            with c1:
                if st.button("🔥 Yes Delete All"):

                    st.session_state.history = []
                    st.session_state.selected_chats.clear()
                    st.session_state.confirm_delete_all = False

                    st.success("All chats deleted")
                    st.rerun()

            with c2:
                if st.button("❌ Cancel All"):
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

    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded_file is not None:
        st.session_state.file_bytes = uploaded_file.read()
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.file_uploaded = False

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

    if not st.session_state.file_uploaded:
        st.info("📄 Upload document to start")

    else:
        st.success("✅ Document ready")

        if st.button("📜 Chat History"):
            st.session_state.view = "history"
            st.rerun()

        if st.button("🔄 Upload New Document"):
            st.session_state.clear()
            st.rerun()

        query = st.chat_input("Ask question...")

        if query:
            with st.spinner("Thinking..."):
                retrieved_results = retrieve(
                    query,
                    st.session_state.index,
                    st.session_state.model,
                    st.session_state.bm25,
                    st.session_state.chunks
                )

                retrieved_chunks = [r[0] for r in retrieved_results]
                context = "\n\n---\n\n".join(retrieved_chunks)

                answer = ask_groq(context, query, st.session_state.history)

                st.session_state.history.append((query, answer))
                st.session_state.last_chat = (query, answer)
                st.session_state.last_retrieval = retrieved_results

                score = confidence_score(
                    retrieved_chunks,
                    answer,
                    st.session_state.model
                )
                st.session_state.last_confidence = score

        if st.session_state.last_chat:
            q, a = st.session_state.last_chat
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(a)

            st.write(f"📊 Confidence: {round(st.session_state.last_confidence * 100, 2)}%")

            st.write("### 🔍 Retrieved Chunks & Cosine Similarity")

            for i, (chunk, sim) in enumerate(st.session_state.last_retrieval):
                with st.expander(f"Chunk {i+1} (Similarity: {round(sim, 3)})"):
                    st.write(chunk)
