import os, re, uuid, requests
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

BONSAI_URL = os.environ.get("BONSAI_URL", "")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
INDEX_NAME = "chemrag"
DIM        = 768
CHUNK_SIZE = 150
OVERLAP    = 20

def get_client():
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from urllib.parse import urlparse
    p = urlparse(BONSAI_URL)
    return OpenSearch(
        hosts=[{"host": p.hostname, "port": p.port or 443}],
        http_auth=(p.username, p.password),
        use_ssl=True, verify_certs=True,
        connection_class=RequestsHttpConnection, timeout=30)

def create_index(client):
    if client.indices.exists(index=INDEX_NAME):
        return
    client.indices.create(index=INDEX_NAME, body={
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "embedding": {"type": "knn_vector", "dimension": DIM,
                         "method": {"name": "hnsw", "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                    "parameters": {"ef_construction": 128, "m": 16}}},
            "text":    {"type": "text"},
            "title":   {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "source":  {"type": "keyword"},
            "doc_id":  {"type": "keyword"},
            "chunk_idx": {"type": "integer"},
        }}})

def embed(texts):
    r = requests.post(
        "https://api-inference.huggingface.co/models/seyonec/ChemBERTa-zinc-base-v1",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": texts, "options": {"wait_for_model": True}}, timeout=60)
    r.raise_for_status()
    embeddings = []
    for item in r.json():
        arr = np.array(item)
        vec = arr.mean(axis=0) if arr.ndim == 2 else arr
        norm = np.linalg.norm(vec)
        embeddings.append((vec / norm if norm > 0 else vec).tolist())
    return embeddings

def chunk_text(text):
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words): break
        start += CHUNK_SIZE - OVERLAP
    return chunks

def ingest_doc(title, text, source, doc_id=None):
    client = get_client()
    create_index(client)
    doc_id = doc_id or f"doc_{uuid.uuid4().hex[:8]}"
    r = client.search(index=INDEX_NAME,
                      body={"query": {"term": {"doc_id": doc_id}}, "size": 1})
    if r["hits"]["total"]["value"] > 0:
        return {"doc_id": doc_id, "chunks": 0, "status": "exists"}
    chunks = chunk_text(text)
    embeddings = embed(chunks)
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        client.index(index=INDEX_NAME, id=str(uuid.uuid4()), body={
            "doc_id": doc_id, "title": title, "source": source,
            "text": chunk, "chunk_idx": i, "embedding": emb})
    client.indices.refresh(index=INDEX_NAME)
    total = client.count(index=INDEX_NAME)["count"]
    return {"doc_id": doc_id, "chunks": len(chunks), "total": total, "status": "ingested"}

def search(query, top_k=5):
    client = get_client()
    qvec = embed([query])[0]
    dense = client.search(index=INDEX_NAME, body={
        "size": top_k*2,
        "query": {"knn": {"embedding": {"vector": qvec, "k": top_k*2}}},
        "_source": ["doc_id","title","source","text"]})["hits"]["hits"]
    sparse = client.search(index=INDEX_NAME, body={
        "size": top_k*2,
        "query": {"multi_match": {"query": query, "fields": ["text^2","title"]}},
        "_source": ["doc_id","title","source","text"]})["hits"]["hits"]
    def norm(hits):
        scores = [h["_score"] for h in hits]
        mn, mx = min(scores, default=0), max(scores, default=1)
        d = mx - mn or 1
        return {h["_source"]["text"][:80]: (h["_source"], (h["_score"]-mn)/d) for h in hits}
    dn, sn = norm(dense), norm(sparse)
    combined = {}
    for k in set(dn) | set(sn):
        dc, ds = dn.get(k, (None, 0))
        sc, ss = sn.get(k, (None, 0))
        combined[k] = (dc or sc, 0.6*ds + 0.4*ss)
    results, seen = [], set()
    for chunk, score in sorted(combined.values(), key=lambda x: x[1], reverse=True):
        if chunk and chunk["doc_id"] not in seen:
            seen.add(chunk["doc_id"])
            results.append((chunk, round(score, 4)))
        if len(results) >= top_k: break
    return results

def generate(question, retrieved):
    context = "\n\n".join(
        f"[Document {i+1}: {c['title']}]\n{c['text']}"
        for i, (c, _) in enumerate(retrieved))
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={"model": "llama-3.1-8b-instant", "temperature": 0.1, "max_tokens": 600,
              "messages": [
                  {"role": "system", "content": "Answer ONLY from context. Cite document titles."},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
              ]}, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

SAMPLES = [
    {"doc_id":"s1","title":"Aqueous Solubility in Drug Discovery","source":"ChemRAG",
     "text":"Aqueous solubility is critical in drug discovery. LogP is the logarithm of the partition coefficient between octanol and water measuring lipophilicity. High LogP means high lipophilicity and poor water solubility. Lipinski rule of five states molecular weight under 500 Da LogP under 5 hydrogen bond donors under 5 acceptors under 10. ChemBERT and MPNN predict solubility with RMSE below 1 log unit. ESOL model uses linear regression on LogP molecular weight and rotatable bonds."},
    {"doc_id":"s2","title":"Retrosynthesis and Synthesis Planning","source":"ChemRAG",
     "text":"Retrosynthesis by E J Corey works backwards from target molecule to precursors. Template free seq2seq networks treat SMILES as sequences. USPTO 50K has 50000 reactions across 10 classes including heteroatom alkylation acylation C C bond formation heterocycle formation protections deprotections reductions oxidations FGI and FGA. Beam search generates candidates ranked by log probability. SMILES augmentation improves generalisation. Reaxys has over 500 million curated reactions."},
    {"doc_id":"s3","title":"ChemBERT and Transformers for Chemistry","source":"ChemRAG",
     "text":"ChemBERT is BERT pre trained on 77 million SMILES strings from PubChem. SMILES tokens include each atom bond bracket as one token. ChemBERT achieves strong performance on solubility toxicity bioactivity prediction. Molecular Transformer by Schwaller applies seq2seq to reaction prediction. Graph Neural Networks represent molecules as graphs atoms as nodes bonds as edges. MPNNs iterate over graph neighbourhood. Multi modal architectures combining SMILES transformers with GNNs achieve best performance."},
    {"doc_id":"s4","title":"Chemical NER and Information Extraction","source":"ChemRAG",
     "text":"Chemical NER identifies compound names formulas SMILES strings in scientific text. ChemBERT MatBERT BioBERT outperform general BERT. BIO tagging uses Beginning Inside Outside labels. Key datasets BC5CDR CHEMDNER NLMChem. Elsevier enrichment pipelines for Reaxys and Embase use NER then relation extraction entity linking and attribute extraction for yield temperature and solvent."},
    {"doc_id":"s5","title":"RAG Retrieval Augmented Generation","source":"ChemRAG",
     "text":"RAG combines retrieval with language model generation for grounded answers. Chunking splits text into overlapping passages. Embedding converts chunks to dense vectors. FAISS IndexFlatIP stores vectors for cosine similarity search. OpenSearch provides HNSW vector search and BM25 keyword search for hybrid retrieval. RAG prevents hallucination by anchoring to retrieved documents. RAGAS evaluates faithfulness answer relevance and context precision."},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client = get_client()
        create_index(client)
        n = client.count(index=INDEX_NAME)["count"]
        print(f"OpenSearch ready | chunks={n}")
    except Exception as e:
        print(f"OpenSearch startup: {e}")
    yield

app = FastAPI(title="ChemRAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class IngestReq(BaseModel):
    title: str; text: str; source: str = "manual"; doc_id: str = None
class QueryReq(BaseModel):
    question: str; top_k: int = 5

@app.get("/")
def root():
    return {"message": "ChemRAG API running", "status": "ok"}

@app.get("/health")
def health():
    try:
        client = get_client()
        n = client.count(index=INDEX_NAME)["count"]
        return {"status": "ok", "chunks": n,
                "embedding": "ChemBERT 768-dim", "retrieval": "HNSW + BM25 hybrid"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/ingest")
def ingest(req: IngestReq):
    try:
        return ingest_doc(req.title, req.text, req.source, req.doc_id)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/ingest/samples")
def ingest_samples():
    results = []
    for doc in SAMPLES:
        try:
            results.append(ingest_doc(doc["title"], doc["text"], doc["source"], doc["doc_id"]))
        except Exception as e:
            results.append({"error": str(e)})
    return {"ingested": len(results), "details": results}

@app.post("/query")
def query(req: QueryReq):
    try:
        retrieved = search(req.question, top_k=req.top_k)
        if not retrieved:
            return {"answer": "No documents found.", "sources": [], "grounded": False}
        answer = generate(req.question, retrieved)
        return {"answer": answer, "grounded": True,
                "retrieval_count": len(retrieved),
                "retrieval_type": "HNSW + BM25 hybrid",
                "sources": [{"title": c["title"], "source": c["source"],
                             "excerpt": c["text"][:300], "score": s}
                            for c, s in retrieved]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/documents")
def documents():
    try:
        client = get_client()
        r = client.search(index=INDEX_NAME, body={
            "size": 0,
            "aggs": {"docs": {"terms": {"field": "doc_id", "size": 1000},
                              "aggs": {"title": {"terms": {"field": "title.keyword", "size": 1}}}}}})
        return [{"doc_id": b["key"],
                 "title": b["title"]["buckets"][0]["key"] if b["title"]["buckets"] else "Unknown",
                 "chunk_count": b["doc_count"]}
                for b in r["aggregations"]["docs"]["buckets"]]
    except Exception as e:
        raise HTTPException(500, str(e))
