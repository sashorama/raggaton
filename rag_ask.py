import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import ollama
import chromadb
import os
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Cached BM25 index — rebuilt when the collection changes
_bm25_cache = {"key": None, "bm25": None, "corpus": None}

def get_bm25(collection):
    key = collection.count()
    if _bm25_cache["key"] != key:
        result = collection.get(include=["documents"])
        corpus = result["documents"]
        tokenized = [doc.lower().split() for doc in corpus]
        _bm25_cache["key"] = key
        _bm25_cache["bm25"] = BM25Okapi(tokenized)
        _bm25_cache["corpus"] = corpus
    return _bm25_cache["bm25"], _bm25_cache["corpus"]

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

# Retrieve and rerank relevant chunks using hybrid search (vector + BM25)
def search(collection, question, top_n=5):
    # --- Vector retrieval ---
    q_emb = ollama.embed(
        model="nomic-embed-text",
        input=question
    ).embeddings[0]

    vec_results = collection.query(
        query_embeddings=[q_emb],
        n_results=20
    )
    vec_ranked = vec_results["documents"][0]

    # --- BM25 retrieval ---
    bm25, corpus = get_bm25(collection)
    tokens = question.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_ranked = [corpus[i] for i in sorted(range(len(bm25_scores)),
                                              key=lambda i: bm25_scores[i],
                                              reverse=True)[:20]]

    # --- Reciprocal Rank Fusion ---
    fused = reciprocal_rank_fusion([vec_ranked, bm25_ranked])
    candidates = sorted(fused, key=fused.get, reverse=True)[:30]

    # --- Cross-encoder reranking ---
    pairs = [[question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]

# Query RAG
def ask(collection, question, chat_model):
    top_docs = search(collection, question)
    context = "\n\n".join(top_docs)

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
        think=False
    )

    return response["message"]["content"]

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
    print("\nCommands: /search <query>  —  return relevant sections only")
    print("          /ask <query>    —  get an answer from the model")
    print("          /quit           —  exit\n")

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
            docs = search(collection, query)
            print("\n" + "=" * 50 + " SEARCH RESULTS " + "=" * 50)
            for i, doc in enumerate(docs, 1):
                print(f"\n--- Result {i} ---")
                print(doc)
            print("=" * 117 + "\n")
        elif raw.startswith("/ask "):
            query = raw[len("/ask "):].strip()
            print("\n" + "=" * 50 + " ANSWER " + "=" * 50)
            print(ask(collection, query, chat_model))
            print("=" * 108 + "\n")
        else:
            print("Unknown command. Use /search <query>, /ask <query>, or /quit.")
