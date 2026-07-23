import sys
import json
import asyncio
import os
# Force HuggingFace to use only local cache — no network calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path
from contextlib import asynccontextmanager

import chromadb
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from rag_ask import search, _build_context

DB_FOLDER = "vector_db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = chromadb.PersistentClient(path=DB_FOLDER)
    yield


app = FastAPI(title="RAG Web UI", lifespan=lifespan)

_static = Path(__file__).parent / "static"
_static.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(_static / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def models():
    try:
        resp = ollama.list()
        return {"models": [m.model for m in resp.models]}
    except Exception as exc:
        raise HTTPException(503, detail=str(exc))


@app.get("/collections")
async def collections():
    cols = app.state.db.list_collections()
    return {"collections": sorted(c.name for c in cols)}


# ── request bodies ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    question: str
    collection: str


class AskRequest(BaseModel):
    question: str
    model: str
    collection: str


class ContextRequest(BaseModel):
    question: str
    collection: str


# ── helpers ───────────────────────────────────────────────────────────────────

def _format_sources(results: list) -> list:
    out = []
    for r in results:
        meta = r.get("metadata", {})
        heading_parts = [h for h in (meta.get("h1", ""), meta.get("h2", ""), meta.get("h3", "")) if h]
        out.append({
            "source":       meta.get("source", "Unknown"),
            "section":      " > ".join(heading_parts),
            "score":        r.get("score"),
            "vec_score":    r.get("vec_score"),
            "bm25_score":   r.get("bm25_score"),
            "sources_type": r.get("sources", "vector"),
            "document":     r.get("document", ""),
        })
    return out


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/search")
async def search_endpoint(req: SearchRequest):
    try:
        col = app.state.db.get_collection(req.collection)
    except Exception:
        raise HTTPException(404, detail=f"Collection '{req.collection}' not found")

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, lambda: search(col, req.question))
    return {"results": _format_sources(results)}


@app.post("/context")
async def context_endpoint(req: ContextRequest):
    try:
        col = app.state.db.get_collection(req.collection)
    except Exception:
        raise HTTPException(404, detail=f"Collection '{req.collection}' not found")

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, lambda: search(col, req.question))
    context = _build_context(results)
    return {"context": context}


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    try:
        col = app.state.db.get_collection(req.collection)
    except Exception:
        raise HTTPException(404, detail=f"Collection '{req.collection}' not found")

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, lambda: search(col, req.question))
    context = _build_context(results)
    sources = _format_sources(results)

    async def generate():
        # Send sources immediately so the UI can render them while the LLM streams
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"

        q: asyncio.Queue = asyncio.Queue()

        def _stream():
            try:
                stream = ollama.chat(
                    model=req.model,
                    messages=[
                        {"role": "system", "content": "Answer using only the provided context."},
                        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion:\n{req.question}"},
                    ],
                    think=True,
                    stream=True,
                    keep_alive="10m",
                    options={"num_ctx": 16384},
                )
                for chunk in stream:
                    msg = chunk["message"]
                    if msg.get("thinking"):
                        loop.call_soon_threadsafe(
                            q.put_nowait,
                            json.dumps({"type": "thinking", "content": msg["thinking"]}) + "\n",
                        )
                    if msg.get("content"):
                        loop.call_soon_threadsafe(
                            q.put_nowait,
                            json.dumps({"type": "answer", "content": msg["content"]}) + "\n",
                        )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    q.put_nowait,
                    json.dumps({"type": "error", "content": str(exc)}) + "\n",
                )
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

        loop.run_in_executor(None, _stream)

        while True:
            item = await q.get()
            if item is None:
                break
            yield item

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_server:app", host="0.0.0.0", port=8000, reload=False)
