import os, requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="ChemRAG", page_icon="🧪",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.answer-box { background:#f0fff4; border:2px solid #2d6a4f;
              border-radius:10px; padding:20px; margin:12px 0;
              font-size:0.95rem; line-height:1.8; color:#1a202c; }
section[data-testid="stSidebar"] { background:#0a1628; }
section[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
</style>""", unsafe_allow_html=True)

# API base — from secrets or env
try:
    API_BASE = st.secrets["API_BASE"]
except Exception:
    API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post(path, data):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

EXAMPLES = [
    "What is the relationship between LogP and drug solubility?",
    "How does retrosynthesis work and what is USPTO 50K?",
    "Explain how ChemBERT represents molecular structures",
    "How does RAG prevent hallucination in LLMs?",
    "What reaction classes are in the USPTO 50K dataset?",
    "Explain Lipinski rule of five for drug discovery",
    "How does hybrid retrieval combine HNSW and BM25?",
    "What is chemical named entity recognition?",
    "How does SMILES augmentation improve retrosynthesis?",
    "What is beam search in synthesis prediction?",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:2.4rem'>🧪</div>
        <div style='font-size:1.1rem;font-weight:700'>ChemRAG</div>
        <div style='font-size:0.7rem;opacity:0.5'>Production · Dr. Mushtaq Ali · KIT</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Home", "💬 Ask ChemRAG",
        "📥 Ingest", "📚 Library", "🔬 Architecture",
    ], label_visibility="collapsed")

    st.divider()

    health = api_get("/health")
    if "error" not in health:
        st.markdown(f"**Docs:** {health.get('document_count', 0)}  |  **Chunks:** {health.get('chunks', 0)}")
        st.markdown("🟢 **API online**")
        st.markdown("🔵 **ChemBERT 768-dim**")
        st.markdown("🟣 **HNSW + BM25 hybrid**")
    else:
        st.error("API offline")
        if st.button("Wake up API"):
            with st.spinner("Waking up Render (30-60 sec)..."):
                api_get("/health")
            st.rerun()

    st.divider()
    top_k = st.slider("Retrieve k chunks", 1, 10, 5)

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🧪 ChemRAG — Production RAG System")
    st.caption("ChemBERT 768-dim · OpenSearch HNSW + BM25 hybrid · Groq LLM · Dr. Mushtaq Ali · KIT")

    health = api_get("/health")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks",       health.get("chunks", 0))
    c2.metric("Embedding",    "ChemBERT")
    c3.metric("Index",        "HNSW")
    c4.metric("Retrieval",    "Hybrid")

    st.divider()
    cl, cr = st.columns(2)
    with cl:
        st.subheader("Quick Start")
        st.markdown("""
        1. Go to **📥 Ingest** → **Load Sample Docs**
        2. Go to **💬 Ask ChemRAG** → ask a question
        """)
    with cr:
        st.subheader("Try these")
        for ex in EXAMPLES[:5]:
            if st.button(ex[:55], key=f"h_{ex[:15]}"):
                st.session_state["pq"] = ex
                st.rerun()

# ── ASK ───────────────────────────────────────────────────────────────────────
elif page == "💬 Ask ChemRAG":
    st.header("💬 Ask ChemRAG")

    default_q = st.session_state.pop("pq", st.session_state.get("lq", ""))
    question  = st.text_input(
        "Your chemistry question", value=default_q,
        placeholder="e.g. What is LogP and how does it affect solubility?")

    ca, cb = st.columns([3, 1])
    with ca:
        ask = st.button("🔍 Ask ChemRAG", type="primary", use_container_width=True)
    with cb:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("result", None)
            st.rerun()

    st.caption("Quick examples:")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES[:6]):
        with cols[i % 3]:
            if st.button(ex[:44], key=f"ex_{i}", use_container_width=True):
                st.session_state["pq"] = ex
                st.rerun()

    if ask and question.strip():
        st.session_state["lq"] = question.strip()
        with st.spinner("ChemBERT embedding → OpenSearch hybrid search → Groq generation..."):
            result = api_post("/query", {"question": question.strip(), "top_k": top_k})
            if "error" not in result:
                st.session_state["result"] = result
            else:
                st.error(f"Error: {result['error']}")

    result = st.session_state.get("result")
    if not result:
        st.stop()

    st.divider()

    grounded = result.get("grounded", False)
    if not grounded:
        st.warning(result.get("answer", "No answer"))
        st.stop()

    st.info(f"✅ {result.get('retrieval_count', 0)} chunks retrieved · {result.get('retrieval_type', 'hybrid')}")

    answer_html = result["answer"].replace("\n", "<br>")
    st.markdown(f'<div class="answer-box">{answer_html}</div>', unsafe_allow_html=True)

    sources = result.get("sources", [])
    if sources:
        st.subheader(f"📚 Sources ({len(sources)})")
        if len(sources) > 1:
            fig = go.Figure(go.Bar(
                x=[s["score"] for s in sources],
                y=[s["title"][:35] for s in sources],
                orientation="h",
                marker_color=["#2d6a4f" if i == 0 else "#68d391" for i in range(len(sources))],
                text=[f"{s['score']:.3f}" for s in sources],
                textposition="outside",
            ))
            fig.update_layout(
                title="Hybrid Scores (HNSW dense + BM25 sparse)",
                height=max(150, len(sources) * 45),
                margin=dict(l=0, r=60, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        for s in sources:
            with st.expander(f"📄 {s['title']}  (score: {s['score']:.4f})"):
                st.write(f"**Source:** {s['source']}")
                st.write(s["excerpt"])

# ── INGEST ────────────────────────────────────────────────────────────────────
elif page == "📥 Ingest":
    st.header("📥 Ingest Documents")
    tab1, tab2, tab3 = st.tabs(["📦 Sample Docs", "📝 Paste Text", "📄 Upload PDF"])

    with tab1:
        st.markdown("Load 5 built-in chemistry documents.")
        if st.button("⚡ Load Sample Documents", type="primary"):
            with st.spinner("ChemBERT embedding → Bonsai OpenSearch indexing..."):
                r = api_post("/ingest/samples", {})
                if "error" not in r:
                    st.success(f"Ingested {r.get('ingested', 0)} documents")
                else:
                    st.error(f"Failed: {r['error']}")
            st.rerun()

    with tab2:
        title  = st.text_input("Title")
        source = st.text_input("Source", value="manual")
        text   = st.text_area("Text", height=200)
        if st.button("📝 Ingest", type="primary"):
            if title and text:
                with st.spinner("Ingesting..."):
                    r = api_post("/ingest", {"title": title, "text": text, "source": source})
                    if "error" not in r:
                        st.success(f"Ingested {r.get('chunks', 0)} chunks")
                    else:
                        st.error(r["error"])
                st.rerun()

    with tab3:
        try:
            import fitz
            uploaded = st.file_uploader("Upload PDF", type=["pdf"])
            if uploaded:
                pdf_title  = st.text_input("Title", value=uploaded.name.replace(".pdf", ""))
                pdf_source = st.text_input("DOI", placeholder="https://doi.org/...")
                if st.button("Extract and Ingest", type="primary"):
                    with st.spinner("Extracting and ingesting..."):
                        pdf_bytes = uploaded.read()
                        doc       = fitz.open(stream=pdf_bytes, filetype="pdf")
                        pages     = [doc[i].get_text() for i in range(len(doc))]
                        doc.close()
                        full_text = " ".join(" ".join(pages).split())
                        if len(full_text.split()) < 50:
                            st.error("Too little text extracted")
                        else:
                            r = api_post("/ingest", {"title": pdf_title,
                                                      "text": full_text,
                                                      "source": pdf_source or uploaded.name})
                            if "error" not in r:
                                st.success(f"Ingested {r.get('chunks', 0)} chunks")
                            else:
                                st.error(r["error"])
                    st.rerun()
        except ImportError:
            st.warning("Run: pip install pymupdf")

# ── LIBRARY ───────────────────────────────────────────────────────────────────
elif page == "📚 Library":
    st.header("📚 Document Library")
    docs = api_get("/documents")
    if isinstance(docs, list) and docs:
        st.metric("Documents in Bonsai OpenSearch", len(docs))
        for doc in docs:
            with st.expander(f"📄 {doc['title']}  ({doc['chunk_count']} chunks)"):
                st.write(f"**Doc ID:** `{doc['doc_id']}`")
    else:
        st.info("No documents yet — go to Ingest")

# ── ARCHITECTURE ──────────────────────────────────────────────────────────────
elif page == "🔬 Architecture":
    st.header("🔬 Production Architecture")
    for title, body in [
        ("① ChemBERT Embedding",
         "seyonec/ChemBERTa-zinc-base-v1 via HuggingFace Inference API. 768-dim vectors. "
         "Pre-trained on 77M SMILES from PubChem. Better chemistry vocabulary than all-MiniLM."),
        ("② OpenSearch HNSW Index",
         "Bonsai-hosted OpenSearch. knn_vector field with HNSW algorithm. "
         "m=16 connections, ef_construction=128, cosinesimil distance. "
         "Persistent — survives restarts. Scales to billions of vectors."),
        ("③ BM25 Text Index",
         "OpenSearch multi_match on text and title fields. "
         "Exact chemical name matching — ibuprofen matches ibuprofen exactly. "
         "Essential for chemistry queries where compound names must match precisely."),
        ("④ Hybrid Retrieval",
         "0.6 x HNSW dense + 0.4 x BM25 sparse. Both scores normalised to 0-1. "
         "Dense finds semantic similarity, sparse finds exact keywords. "
         "Best of both worlds for chemistry queries."),
        ("⑤ Groq Generation",
         "llama-3.1-8b-instant. Temperature 0.1. Max 600 tokens. "
         "System prompt constrains to context only — prevents hallucination. "
         "Every answer traceable to source document."),
    ]:
        with st.expander(title):
            st.write(body)

    st.subheader("Full Cloud Stack — All Free")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vector DB", "Bonsai", "OpenSearch free")
    c2.metric("API", "Render", "FastAPI free")
    c3.metric("Frontend", "Streamlit", "Cloud free")
    c4.metric("LLM", "Groq", "Free tier")
