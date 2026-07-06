import os, uuid, requests
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

BONSAI_URL = os.environ.get("BONSAI_URL", "")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
INDEX      = "chemrag"
DIM        = 768
CHUNK      = 150
OVERLAP    = 20

def get_os():
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from urllib.parse import urlparse
    p = urlparse(BONSAI_URL)
    return OpenSearch(
        hosts=[{"host": p.hostname, "port": p.port or 443}],
        http_auth=(p.username, p.password),
        use_ssl=True, verify_certs=True,
        connection_class=RequestsHttpConnection, timeout=30)

def ensure_index(os_client):
    if os_client.indices.exists(index=INDEX):
        return
    os_client.indices.create(index=INDEX, body={
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "embedding": {"type": "knn_vector", "dimension": DIM,
                         "method": {"name": "hnsw", "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                    "parameters": {"ef_construction": 128, "m": 16}}},
            "text":   {"type": "text"},
            "title":  {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "source": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "idx":    {"type": "integer"},
        }}})

def embed(texts):
    r = requests.post(
        "https://api-inference.huggingface.co/models/seyonec/ChemBERTa-zinc-base-v1",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60)
    r.raise_for_status()
    out = []
    for item in r.json():
        arr = np.array(item)
        vec = arr.mean(axis=0) if arr.ndim == 2 else arr
        n = np.linalg.norm(vec)
        out.append((vec / n if n > 0 else vec).tolist())
    return out

def chunks(text):
    w, res, s = text.split(), [], 0
    while s < len(w):
        e = min(s + CHUNK, len(w))
        res.append(" ".join(w[s:e]))
        if e == len(w): break
        s += CHUNK - OVERLAP
    return res

def ingest(title, text, source, doc_id=None):
    c = get_os()
    ensure_index(c)
    doc_id = doc_id or f"d_{uuid.uuid4().hex[:8]}"
    r = c.search(index=INDEX, body={"query": {"term": {"doc_id": doc_id}}, "size": 1})
    if r["hits"]["total"]["value"] > 0:
        return {"doc_id": doc_id, "chunks": 0, "status": "exists"}
    ch = chunks(text)
    em = embed(ch)
    for i, (chunk, emb) in enumerate(zip(ch, em)):
        c.index(index=INDEX, id=str(uuid.uuid4()), body={
            "doc_id": doc_id, "title": title, "source": source,
            "text": chunk, "idx": i, "embedding": emb})
    c.indices.refresh(index=INDEX)
    return {"doc_id": doc_id, "chunks": len(ch),
            "total": c.count(index=INDEX)["count"], "status": "ok"}

def search(query, k=5):
    c = get_os()
    qv = embed([query])[0]
    d = c.search(index=INDEX, body={
        "size": k*2,
        "query": {"knn": {"embedding": {"vector": qv, "k": k*2}}},
        "_source": ["doc_id","title","source","text"]})["hits"]["hits"]
    s = c.search(index=INDEX, body={
        "size": k*2,
        "query": {"multi_match": {"query": query, "fields": ["text^2","title"]}},
        "_source": ["doc_id","title","source","text"]})["hits"]["hits"]
    def norm(hits):
        sc = [h["_score"] for h in hits]
        mn, mx = min(sc, default=0), max(sc, default=1)
        dd = mx - mn or 1
        return {h["_source"]["text"][:80]: (h["_source"], (h["_score"]-mn)/dd) for h in hits}
    dn, sn = norm(d), norm(s)
    combined = {}
    for key in set(dn)|set(sn):
        dc, ds = dn.get(key, (None, 0))
        sc2, ss = sn.get(key, (None, 0))
        combined[key] = (dc or sc2, 0.6*ds + 0.4*ss)
    res, seen = [], set()
    for chunk, score in sorted(combined.values(), key=lambda x: x[1], reverse=True):
        if chunk and chunk["doc_id"] not in seen:
            seen.add(chunk["doc_id"])
            res.append((chunk, round(score, 4)))
        if len(res) >= k: break
    return res

def generate(q, retrieved):
    ctx = "\n\n".join(
        f"[Doc {i+1}: {c['title']}]\n{c['text']}"
        for i, (c, _) in enumerate(retrieved))
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}",
                 "Content-Type": "application/json"},
        json={"model": "llama-3.1-8b-instant", "temperature": 0.1, "max_tokens": 600,
              "messages": [
                  {"role": "system", "content": "Answer ONLY from context. Cite doc titles."},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}
              ]}, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

SAMPLES = [
    {"id":"s1","title":"Aqueous Solubility in Drug Discovery","src":"ChemRAG",
     "text":"Aqueous solubility is critical in drug discovery. LogP is lipophilicity partition coefficient between octanol and water. High LogP means poor water solubility. Lipinski rule of five states molecular weight under 500 Da LogP under 5 hydrogen bond donors under 5 acceptors under 10. ChemBERT MPNN predict solubility RMSE below 1 log unit. ESOL model uses linear regression on LogP molecular weight rotatable bonds."},
    {"id":"s2","title":"Retrosynthesis and Synthesis Planning","src":"ChemRAG",
     "text":"Retrosynthesis by Corey works backwards from target molecule to precursors. Template free seq2seq networks treat SMILES as sequences. USPTO 50K has 50000 reactions across 10 classes: heteroatom alkylation acylation C-C bond formation heterocycle formation protections deprotections reductions oxidations FGI FGA. Beam search generates candidates ranked by log probability. SMILES augmentation improves generalisation. Reaxys has 500 million curated reactions."},
    {"id":"s3","title":"ChemBERT and Transformers","src":"ChemRAG",
     "text":"ChemBERT is BERT pre-trained on 77 million SMILES from PubChem. SMILES tokens are atoms bonds brackets. ChemBERT achieves strong performance on solubility toxicity bioactivity. Molecular Transformer applies seq2seq to reaction prediction. Graph Neural Networks represent molecules as graphs atoms as nodes bonds as edges. MPNNs iterate over graph neighbourhood."},
    {"id":"s4","title":"Chemical NER and Information Extraction","src":"ChemRAG",
     "text":"Chemical NER identifies compound names formulas SMILES in scientific text. BIO tagging uses Beginning Inside Outside labels. Key datasets BC5CDR CHEMDNER NLMChem. Elsevier enrichment pipelines for Reaxys Embase use NER then relation extraction entity linking attribute extraction for yield temperature solvent."},
    {"id":"s5","title":"RAG Retrieval Augmented Generation","src":"ChemRAG",
     "text":"RAG combines retrieval with language model generation for grounded answers. Chunking splits text into overlapping passages. Embedding converts chunks to dense vectors. FAISS HNSW store vectors for cosine similarity search. OpenSearch provides HNSW vector search and BM25 keyword search for hybrid retrieval. RAG prevents hallucination by anchoring to retrieved documents. RAGAS evaluates faithfulness answer relevance context precision."},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        c = get_os()
        ensure_index(c)
        n = c.count(index=INDEX)["count"]
        print(f"Ready | chunks={n}")
    except Exception as e:
        print(f"OpenSearch warning: {e}")
    yield

app = FastAPI(title="ChemRAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class IR(BaseModel):
    title: str; text: str; source: str = "manual"; doc_id: str = None
class QR(BaseModel):
    question: str; top_k: int = 5

@app.get("/")
def root():
    return {"status": "ok", "service": "ChemRAG API"}

@app.get("/health")
def health():
    try:
        c = get_os()
        n = c.count(index=INDEX)["count"]
        return {"status": "ok", "chunks": n,
                "embedding": "ChemBERT 768-dim",
                "retrieval": "HNSW + BM25 hybrid"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/ingest")
def ingest_ep(r: IR):
    try:
        return ingest(r.title, r.text, r.source, r.doc_id)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/ingest/samples")
@app.post("/ingest/samples")
def ingest_samples():
    res = []
    errors = []
    for doc in SAMPLES:
        try:
            r = ingest(doc["title"], doc["text"], doc["src"], doc["id"])
            res.append(r)
        except Exception as e:
            errors.append({"doc": doc["title"], "error": str(e)})
    try:
        c = get_os()
        total = c.count(index=INDEX)["count"]
    except Exception as e:
        total = -1
        errors.append({"count_error": str(e)})
    return {
        "ingested": len(res),
        "total_chunks": total,
        "results": res,
        "errors": errors,
    }

@app.get("/ingest/test")
def ingest_test():
    """Test endpoint to debug ingestion step by step."""
    steps = {}
    # Step 1: OpenSearch connection
    try:
        c = get_os()
        steps["opensearch"] = "connected"
    except Exception as e:
        return {"failed_at": "opensearch", "error": str(e), "steps": steps}
    # Step 2: Index creation
    try:
        ensure_index(c)
        steps["index"] = "ready"
    except Exception as e:
        return {"failed_at": "index", "error": str(e), "steps": steps}
    # Step 3: HuggingFace embedding
    try:
        test_vec = embed(["test chemistry text"])
        steps["embedding"] = f"ok - dim={len(test_vec[0])}"
    except Exception as e:
        return {"failed_at": "embedding", "error": str(e), "steps": steps}
    # Step 4: Index one chunk
    try:
        import uuid as _uuid
        c.index(index=INDEX, id=str(_uuid.uuid4()), body={
            "doc_id": "test_doc", "title": "Test", "source": "test",
            "text": "test chemistry text", "idx": 0,
            "embedding": test_vec[0]})
        c.indices.refresh(index=INDEX)
        n = c.count(index=INDEX)["count"]
        steps["index_chunk"] = f"ok - total={n}"
    except Exception as e:
        return {"failed_at": "indexing", "error": str(e), "steps": steps}
    return {"status": "all steps passed", "steps": steps}

@app.post("/query")
def query_ep(r: QR):
    try:
        retrieved = search(r.question, k=r.top_k)
        if not retrieved:
            return {"answer": "No documents found.", "sources": [], "grounded": False}
        answer = generate(r.question, retrieved)
        return {"answer": answer, "grounded": True,
                "retrieval_count": len(retrieved),
                "retrieval_type": "HNSW + BM25 hybrid",
                "sources": [{"title": c["title"], "source": c["source"],
                             "excerpt": c["text"][:300], "score": sc}
                            for c, sc in retrieved]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/documents")
def docs_ep():
    try:
        c = get_os()
        r = c.search(index=INDEX, body={
            "size": 0,
            "aggs": {"docs": {"terms": {"field": "doc_id", "size": 100},
                              "aggs": {"t": {"terms": {"field": "title.keyword", "size": 1}}}}}})
        return [{"doc_id": b["key"],
                 "title": b["t"]["buckets"][0]["key"] if b["t"]["buckets"] else "Unknown",
                 "chunk_count": b["doc_count"]}
                for b in r["aggregations"]["docs"]["buckets"]]
    except Exception as e:
        raise HTTPException(500, str(e))
