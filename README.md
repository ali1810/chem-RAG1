# 🧪 ChemRAG — Chemical Literature RAG Assistant

**Live Demo:** [chemrag1.streamlit.app](https://chemrag1.streamlit.app)  
**API:** [ali1810-chemrag-api.hf.space](https://ali1810-chemrag-api.hf.space)  
**Built by:** Dr. Mushtaq Ali | KIT | [github.com/ali1810](https://github.com/ali1810)

---

## What is ChemRAG?

ChemRAG is a production-grade **Retrieval-Augmented Generation (RAG)** system for chemistry literature. Upload any chemistry paper or research document and ask questions about it — every answer is **grounded** and **cited** back to the source document.

Built to demonstrate the NLP enrichment pipeline architecture used in databases like **Reaxys** and **Embase**.

---

## How It Works

```
Your Question
      ↓
① Embed query (all-MiniLM-L6-v2, 384-dim)
      ↓
② OpenSearch Hybrid Retrieval
   ├── HNSW dense search (semantic similarity)
   └── BM25 sparse search (exact keyword match)
      ↓
③ Context Assembly (top-k chunks + prompt engineering)
      ↓
④ Groq LLM Generation (llama-3.1-8b-instant)
      ↓
Grounded answer with citations
```

---

## Features

- **📄 PDF Upload** — upload any chemistry paper and query it instantly
- **🔍 Hybrid Retrieval** — HNSW vector search + BM25 keyword search combined
- **✅ Grounded Answers** — LLM constrained to answer only from retrieved context
- **📊 RAGAS Evaluation** — measure faithfulness, answer relevance, context precision
- **📚 Document Library** — see all ingested documents and chunk counts
- **💬 Example Questions** — quick-start chemistry questions built in

---

## Tech Stack

| Component | Technology |
|---|---|
| **Embedding** | sentence-transformers all-MiniLM-L6-v2 (384-dim) |
| **Vector Index** | OpenSearch HNSW (cosine similarity) |
| **Keyword Search** | OpenSearch BM25 |
| **Retrieval** | Hybrid (0.6 × dense + 0.4 × sparse) |
| **LLM** | Groq llama-3.1-8b-instant |
| **Backend** | FastAPI (HuggingFace Spaces) |
| **Frontend** | Streamlit Cloud |
| **Vector DB** | Bonsai OpenSearch (cloud) |

---

## Why Hybrid Retrieval?

Chemistry queries need both:

- **Dense search** — finds semantically similar content even without exact keywords
  - Query: *"lipophilicity"* → finds chunks about *"LogP"*
- **Sparse search** — finds exact chemical name matches
  - Query: *"ibuprofen"* → must match *"ibuprofen"* exactly

Combined hybrid retrieval handles both cases better than either alone.

---

## Sample Questions

After loading sample documents try:

```
What is the relationship between LogP and drug solubility?
What are the 10 reaction classes in USPTO 50K?
How does RAG prevent hallucination in LLMs?
Explain Lipinski rule of five for drug discovery
How does HNSW perform vector similarity search?
What is ChemBERT and how was it trained?
How does hybrid retrieval combine HNSW and BM25?
```

After uploading a research paper try:

```
What is the main contribution of this paper?
What dataset was used and how many compounds?
What performance metrics were reported?
What are the limitations acknowledged by the authors?
How does this compare to state of the art?
What future work do the authors suggest?
```

---

## Production Architecture

```
User (browser)
      ↓
Streamlit Cloud (frontend)
https://chemrag1.streamlit.app
      ↓ HTTP REST API
HuggingFace Spaces (FastAPI backend)
https://ali1810-chemrag-api.hf.space
      ├── Embed → all-MiniLM-L6-v2 (local, no API needed)
      ├── Search → Bonsai OpenSearch (HNSW + BM25)
      └── Generate → Groq API (llama-3.1-8b-instant)
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check + chunk count |
| `/ingest/samples` | GET/POST | Load built-in chemistry documents |
| `/ingest` | POST | Ingest custom document |
| `/query` | POST | RAG query |
| `/documents` | GET | List all documents |
| `/reset` | GET | Reset OpenSearch index |
| `/docs` | GET | Interactive API documentation |

---

## Local Setup

```bash
# Clone
git clone https://github.com/ali1810/chemrag-api
cd chemrag-api

# Install
pip install fastapi uvicorn opensearch-py sentence-transformers requests numpy

# Environment
export BONSAI_URL="https://user:pass@cluster.bonsai.io"
export GROQ_API_KEY="gsk_your_key"

# Run API
uvicorn main:app --reload --port 8000

# Run Streamlit (new terminal)
pip install streamlit plotly pymupdf
streamlit run streamlit_app.py
```

---

## Research Context

ChemRAG was built as part of my PhD research at **Karlsruhe Institute of Technology (KIT)** to demonstrate:

1. **Production RAG pipeline** — chunking, embedding, vector indexing, hybrid retrieval, grounded generation
2. **Chemistry NLP** — domain-specific document understanding for scientific literature
3. **Full-stack ML deployment** — FastAPI + OpenSearch + Streamlit on free cloud infrastructure

**Related work:**
- Aqueous solubility prediction — [ACS JCIM 2025](https://doi.org/10.1021/acs.jcim.4c02399)
- Retrosynthesis prediction — [HuggingFace](https://huggingface.co/ali1810/retrosynthesis-opennmt)
- ChemPredict platform — [GitHub](https://github.com/ali1810/chempredict)

---

## Author

**Dr. Mushtaq Ali**  
PhD, Karlsruhe Institute of Technology (KIT)  
Computational Chemistry & Machine Learning  
ali10786@gmail.com | [github.com/ali1810](https://github.com/ali1810) | [ACS JCIM 2025](https://doi.org/10.1021/acs.jcim.4c02399)
