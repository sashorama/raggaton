import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import os
# Force HuggingFace to use only local cache — no network calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import ollama
import chromadb
import logging
import readline
import atexit
from pathlib import Path
from datetime import datetime
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# Setup logging
logging.basicConfig(
    filename="ask.log",
    level=logging.INFO,
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Suppress verbose logging from external libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)

# Cached BM25 index — rebuilt when the collection changes or tokenizer version bumps
_TOKENIZER_VERSION = 2
_bm25_cache = {"key": None, "bm25": None, "corpus": None, "corpus_meta": None, "corpus_ids": None}

def _tokenize(text):
    """Lowercase and extract word tokens. Hyphens within a word are kept
    (e.g. 'IOM-s' -> ['iom-s']) so product names stay as single tokens.
    Slashes and backslashes still split (e.g. 'path/to' -> ['path', 'to'])."""
    return re.findall(r"\b\w+(?:-\w+)*\b", text.lower())

def get_bm25(collection):
    key = (collection.count(), _TOKENIZER_VERSION)
    if _bm25_cache["key"] == key:
        return (
            _bm25_cache["bm25"],
            _bm25_cache["corpus"],
            _bm25_cache["corpus_meta"],
            _bm25_cache["corpus_ids"],
        )

    results = collection.get(
        include=["documents", "metadatas"]
    )

    corpus = results["documents"]
    corpus_meta = results["metadatas"]
    corpus_ids = results["ids"]

    tokenized = [_tokenize(doc) for doc in corpus]

    bm25 = BM25Okapi(tokenized)

    _bm25_cache["key"] = key
    _bm25_cache["bm25"] = bm25
    _bm25_cache["corpus"] = corpus
    _bm25_cache["corpus_meta"] = corpus_meta
    _bm25_cache["corpus_ids"] = corpus_ids

    return bm25, corpus, corpus_meta, corpus_ids

def reciprocal_rank_fusion(rankings, k=60):
    """Merge multiple ranked lists of docs into a single scored dict."""
    scores = {}
    for ranked_list in rankings:
        for rank, doc in enumerate(ranked_list):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
    return scores

# Get available models from Ollama
def get_available_models():
    try:
        response = ollama.list()
        models = [model.model for model in response.models]
        return models
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

# Select model from available options
def select_model(model_type):
    models = get_available_models()
    
    if not models:
        print("No models found. Please install models in Ollama.")
        return None
    
    print(f"\nAvailable models for {model_type}:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = input(f"Select {model_type} model (enter number): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def search_bm25(collection, question, top_n=10):
    bm25, corpus, corpus_meta, corpus_ids = get_bm25(collection)

    tokens = _tokenize(question)
    bm25_scores = bm25.get_scores(tokens)

    results = []
    for i in sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_n]:
        if bm25_scores[i] <= 0:
            break
        results.append({
            "document": corpus[i],
            "metadata": corpus_meta[i],
            "bm25_score": round(float(bm25_scores[i]), 4),
        })

    return results


# Retrieve and rerank relevant chunks using hybrid search (vector + BM25)
def search(collection, question, top_n=10):
    # ---------------------------------------------------------
    # 1. Create query embedding
    # ---------------------------------------------------------
    q_emb = ollama.embed(
        model="nomic-embed-text",
        input=question
    ).embeddings[0]


    # ---------------------------------------------------------
    # 2. Vector retrieval from ChromaDB
    # ---------------------------------------------------------
    vec_results = collection.query(
        query_embeddings=[q_emb],
        n_results=30,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    vec_ids = vec_results["ids"][0]
    vec_docs = vec_results["documents"][0]
    vec_meta = vec_results["metadatas"][0]
    vec_distances = vec_results["distances"][0]

    # ID-keyed lookups
    id_to_doc  = {id_: doc  for id_, doc  in zip(vec_ids, vec_docs)}
    id_to_meta = {id_: meta for id_, meta in zip(vec_ids, vec_meta)}

    # source_info[id] = {"vec_score": float|None, "bm25_score": float|None}
    source_info = {
        id_: {"vec_score": round(1 - dist, 4), "bm25_score": None}
        for id_, dist in zip(vec_ids, vec_distances)
    }

    # ---------------------------------------------------------
    # 3. BM25 retrieval
    # ---------------------------------------------------------
    bm25, corpus, corpus_meta, corpus_ids = get_bm25(collection)

    tokens = _tokenize(question)

    bm25_scores = bm25.get_scores(tokens)

    bm25_ids = []

    for i in sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:20]:
        if bm25_scores[i] <= 0:
            break  # remaining scores are also 0 — no token match
        id_ = corpus_ids[i]
        bm25_ids.append(id_)

        id_to_doc[id_]  = corpus[i]
        id_to_meta[id_] = corpus_meta[i]

        bm25_score = round(float(bm25_scores[i]), 4)
        if id_ in source_info:
            source_info[id_]["bm25_score"] = bm25_score
        else:
            source_info[id_] = {"vec_score": None, "bm25_score": bm25_score}
    

    # ---------------------------------------------------------
    # 4. Reciprocal Rank Fusion
    # ---------------------------------------------------------
    fused = reciprocal_rank_fusion(
        [
            vec_ids,
            bm25_ids
        ]
    )

    candidates = sorted(
        fused,
        key=fused.get,
        reverse=True
    )[:30]


    # ---------------------------------------------------------
    # 5. Cross encoder reranking
    # ---------------------------------------------------------
    pairs = [
        [question, id_to_doc[id_]]
        for id_ in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )


    # ---------------------------------------------------------
    # 6. Return documents + metadata
    # ---------------------------------------------------------
    results = []

    for score, id_ in ranked[:top_n]:

        metadata = id_to_meta.get(id_, {})
        info = source_info.get(id_, {"vec_score": None, "bm25_score": None})

        if info["vec_score"] is not None and info["bm25_score"] is not None:
            sources = "vector + bm25"
        elif info["vec_score"] is not None:
            sources = "vector"
        else:
            sources = "bm25"

        results.append(
            {
                "document": id_to_doc.get(id_, ""),
                "metadata": metadata,
                "score": float(score),
                "sources": sources,
                "vec_score": info["vec_score"],
                "bm25_score": info["bm25_score"],
            }
        )

    return results

# Query RAG
def _build_context(top_chunks_with_meta):
    """Group chunks by document then section. Each document and section
    appears exactly once, in order of first appearance in search results.
    Section header shows only the parent levels:
    - h3 chunk → "h1 > h2"
    - h2 chunk → "h1"
    - h1 chunk → no section header
    """
    SEP = "=" * 50
    from collections import OrderedDict

    # docs[source][section_key] = [chunks...]
    # OrderedDict preserves insertion order → first-appearance order
    docs = OrderedDict()

    for doc in top_chunks_with_meta:
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "Unknown")
        h1 = metadata.get("h1", "")
        h2 = metadata.get("h2", "")
        h3 = metadata.get("h3", "")

        # section_key = the heading levels that will be DISPLAYED in the header.
        # h3 chunk → show "h1 > h2" → key = (h1, h2)
        # h2 chunk → show "h1"      → key = (h1,)
        # h1 chunk → show "h1"      → key = (h1,)  ← same bucket as h2 chunks under same h1
        if h3:
            section_key = (h1, h2)
        else:
            section_key = (h1,) if h1 else ()

        if source not in docs:
            docs[source] = OrderedDict()
        if section_key not in docs[source]:
            docs[source][section_key] = []

        docs[source][section_key].append(doc)
    context_parts = []
    
    for source, sections in docs.items():
        doc_lines = [SEP, f"Document: {source}"]
        for section_key, chunks in sections.items():
            if section_key:
                doc_lines.append(f"\nSection: {' > '.join(section_key)}")
            for chunk in chunks:
                doc_lines.append(chunk.get("document", ""))
            
        context_parts.append("\n".join(doc_lines))
    
    return "\n\n".join(context_parts)


def ask(collection, question, chat_model):
    top_docs_with_meta = search(collection, question)
    
    context = _build_context(top_docs_with_meta)

    # Log the question with context
    timestamp_q = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_q = f"\n{'='*80}\nTimestamp: {timestamp_q}\nModel: {chat_model}\n{'='*80}\n\nQUESTION:\n{question}\n\n{'-'*80}\nCONTEXT:\n{'-'*80}\n{context}\n"
    logging.info(log_q)

    response = ollama.chat(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": "Answer using only the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        think=True,
        stream= True,
        keep_alive="10m",
        options={
        "num_ctx": 16384
        }
    )

    #answer = response["message"]["content"]
    
    # Log the answer with separate timestamp
    #timestamp_a = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #log_a = f"\n{'-'*80}\nANSWER (received at {timestamp_a}):\n{'-'*80}\n{answer}\n"
    #logging.info(log_a)

    return response

# Load and query
if __name__ == "__main__":
    db_folder = "vector_db"
    
    # Check if vector DB exists
    if not os.path.exists(db_folder):
        print(f"Vector database not found in '{db_folder}/' folder.")
        print("Please run 'rag_build.py' first to build the database.")
        exit()
    
    print("=" * 50)
    print("RAG Query Interface")
    print("=" * 50)

    # Select collection
    client = chromadb.PersistentClient(path=db_folder)
    collections = client.list_collections()
    names = sorted(c.name for c in collections)

    if not names:
        print("No databases found. Run 'rag_build.py' first.")
        exit()

    collection_name = None
    if len(names) == 1:
        collection_name = names[0]
        print(f"Using database: '{collection_name}'")
    else:
        print("Available databases:")
        for i, name in enumerate(names, 1):
            print(f"  {i}. {name}")
        while collection_name is None:
            try:
                choice = input("Select database (enter number): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(names):
                    collection_name = names[idx]
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a number.")

    # Load persistent collection
    print(f"Loading '{collection_name}'...")
    collection = client.get_collection(collection_name)
    
    count = collection.count()
    print(f"✓ Loaded! {count} embeddings ready.")

    # List indexed documents from DB metadata
    results = collection.get(include=["metadatas"])
    sources = sorted({m["source"] for m in results["metadatas"]}) if results["metadatas"] else []
    if sources:
        print(f"✓ Indexed documents ({len(sources)}):")
        for f in sources:
            print(f"    - {f}")
    
    # Select chat model
    chat_model = select_model("chat")
    if not chat_model:
        exit()
    
    print(f"✓ Selected chat model: {chat_model}")
    print("\nCommands: /search <query>   —  hybrid search (vector + BM25 + rerank)")
    print("          /bm25 <query>    —  keyword search only (raw BM25, no reranking)")
    print("          /context <query>  —  show the context that would be sent to the model")
    print("          /ask <query>     —  get an answer from the model")
    print("          /quit            —  exit\n")
    
    # Setup readline history (up arrow to access previous commands)
    history_file = Path.home() / ".rag_ask_history"
    if history_file.exists():
        readline.read_history_file(history_file)
    readline.set_history_length(100)
    atexit.register(readline.write_history_file, history_file)

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        if raw == "/quit":
            print("Goodbye!")
            break
        elif raw.startswith("/search "):
            query = raw[len("/search "):].strip()
            docs_with_meta = search(collection, query)
            print("\n" + "=" * 50 + " SEARCH RESULTS " + "=" * 50)

            
            for i, result_i in enumerate(docs_with_meta, 1):
                metadata = result_i.get("metadata")
                heading_path = metadata.get("h1",""), metadata.get("h2",""), metadata.get("h3","")
                source = metadata.get("source", "Unknown")
                #print(source)
                #print(heading_path)
               
                # Build retrieval provenance string
                sources = result_i.get("sources", "unknown")
                vec_score = result_i.get("vec_score")
                bm25_score = result_i.get("bm25_score")
                rerank_score = result_i.get("score")

                score_parts = []
                if vec_score is not None:
                    score_parts.append(f"vec={vec_score:.4f}")
                if bm25_score is not None:
                    score_parts.append(f"bm25={bm25_score:.4f}")
                score_parts.append(f"rerank={rerank_score:.4f}")
                score_str = " | ".join(score_parts)

                # Display with hierarchical headings
                print(f"\n--- Result {i} [{sources}] ({score_str}) ---")
                if any(heading_path):
                    heading_str = " > ".join(h for h in heading_path if h)
                    print(f"    Source: {source} | Section: {heading_str}")
                else:
                    print(f"    Source: {source}")
                print(result_i.get("document", ""))
            
            print("=" * 117 + "\n")
        elif raw.startswith("/bm25 "):
            query = raw[len("/bm25 "):].strip()
            results_bm25 = search_bm25(collection, query)
            print("\n" + "=" * 50 + " BM25 RESULTS " + "=" * 50)
            if not results_bm25:
                print("  No results — no query tokens matched any document.")
            for i, result_i in enumerate(results_bm25, 1):
                metadata = result_i.get("metadata", {})
                heading_path = metadata.get("h1",""), metadata.get("h2",""), metadata.get("h3","")
                source = metadata.get("source", "Unknown")
                bm25_score = result_i.get("bm25_score")
                if any(heading_path):
                    heading_str = " > ".join(h for h in heading_path if h)
                    print(f"\n--- Result {i} [bm25={bm25_score:.4f}] ---")
                    print(f"    Source: {source} | Section: {heading_str}")
                else:
                    print(f"\n--- Result {i} [bm25={bm25_score:.4f}] ---")
                    print(f"    Source: {source}")
                print(result_i.get("document", ""))
            print("=" * 114 + "\n")
        elif raw.startswith("/context "):
            query = raw[len("/context "):].strip()
            docs_with_meta = search(collection, query)
            context = _build_context(docs_with_meta)
            print("\n" + "=" * 50 + " CONTEXT " + "=" * 50)
            print(context)
            print("=" * 50 + " END CONTEXT " + "=" * 50 + "\n")
        elif raw == "/context":
            print("Usage: /context <query>")
        elif raw.startswith("/ask "):
            query = raw[len("/ask "):].strip()
            stream = ask(collection, query, chat_model)
            answer_started = False
            for chunk in stream:
                msg = chunk["message"]
                if "thinking" in msg and msg["thinking"]:
                    print(msg["thinking"], end="", flush=True)
                if "content" in msg and msg["content"]:
                    if not answer_started:
                        print("\n" + "=" * 50 + " ANSWER " + "=" * 50)
                        answer_started = True
                    print(msg["content"], end="", flush=True)
            print("\n" + "=" * 108 + "\n")
        else:
            print("Unknown command. Use /search <query>, /context <query>, /ask <query>, or /quit.")
