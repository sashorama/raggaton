import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from pypdf import PdfReader
import os

DB_FOLDER = "vector_db"
EMBEDDING_MODEL = "nomic-embed-text"

# Active collection name — changed at runtime by /db
current_db = "docs"

def docs_folder():
    """Each DB keeps its documents in a folder with the same name."""
    return current_db


# --- Document loading ---

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_document(path):
    if path.lower().endswith('.pdf'):
        return load_pdf(path)
    return load_text(path)

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", " "],
        chunk_size=1500,
        chunk_overlap=300
    )
    return splitter.split_text(text)


# --- DB helpers ---

def get_client():
    os.makedirs(DB_FOLDER, exist_ok=True)
    return chromadb.PersistentClient(path=DB_FOLDER)

def open_collection(create=False):
    client = get_client()
    if create:
        try:
            return client.get_collection(current_db)
        except Exception:
            return client.create_collection(current_db)
    return client.get_collection(current_db)

def get_embedded_sources(collection):
    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return []
    return sorted({m["source"] for m in results["metadatas"]})

def embed_file(collection, file_path, source_name):
    text = load_document(file_path)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(model=EMBEDDING_MODEL, input=chunk).embeddings[0]
        collection.add(
            ids=[f"{source_name}__{i}"],
            embeddings=[emb],
            documents=[chunk],
            metadatas=[{"source": source_name}]
        )
    return len(chunks)


# --- Subcommands ---

def cmd_list():
    try:
        collection = open_collection()
    except Exception:
        print("No vector database found. Run 'build' first.")
        return
    sources = get_embedded_sources(collection)
    if not sources:
        print("No documents in the database.")
    else:
        print(f"Documents in the database ({len(sources)}):")
        for s in sources:
            # Count chunks per source
            res = collection.get(where={"source": s}, include=["metadatas"])
            print(f"  - {s}  ({len(res['ids'])} chunks)")

def cmd_add(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return
    source_name = os.path.basename(file_path)
    collection = open_collection(create=True)
    if source_name in get_embedded_sources(collection):
        print(f"'{source_name}' is already in the database. Remove it first to re-embed.")
        return
    print(f"Embedding '{source_name}'...")
    count = embed_file(collection, file_path, source_name)
    print(f"✓ Added {count} chunks from '{source_name}'.")

def cmd_remove(source_name):
    try:
        collection = open_collection()
    except Exception:
        print("No vector database found.")
        return
    results = collection.get(where={"source": source_name}, include=["metadatas"])
    if not results["ids"]:
        print(f"No document named '{source_name}' found in the database.")
        return
    collection.delete(ids=results["ids"])
    print(f"✓ Removed '{source_name}' ({len(results['ids'])} chunks deleted).")

def cmd_build():
    folder = docs_folder()
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created '{folder}/' folder. Add PDF or text files there.")
        return
    files = [f for f in os.listdir(folder) if f.endswith(('.pdf', '.txt', '.md'))]
    if not files:
        print(f"No PDF or text files found in '{folder}/'.")
        return
    collection = open_collection(create=True)
    embedded = get_embedded_sources(collection)
    new_files = [f for f in files if f not in embedded]
    if not new_files:
        print(f"✓ Database up to date! {collection.count()} embeddings, {len(embedded)} documents.")
        return
    print(f"Found {len(new_files)} new file(s) to embed:")
    for f in new_files:
        print(f"  - {f}")
    answer = input("\nAdd these documents to the database? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return
    for file in new_files:
        file_path = os.path.join(folder, file)
        try:
            count = embed_file(collection, file_path, file)
            print(f"  ✓ {file}: {count} chunks")
        except Exception as e:
            print(f"  ✗ {file}: {e}")
    print(f"\n✓ Done! {collection.count()} total embeddings, {len(files)} documents.")


def cmd_db():
    global current_db
    client = get_client()
    collections = client.list_collections()
    names = sorted(c.name for c in collections)

    print(f"\nAvailable databases (active: '{current_db}'):")
    if names:
        for i, name in enumerate(names, 1):
            marker = " *" if name == current_db else ""
            print(f"  {i}. {name}{marker}")
    else:
        print("  (none yet)")
    print(f"  n. Create new database")

    choice = input("\nSelect number or type name to create: ").strip()
    if not choice:
        return
    if choice.lower() == "n":
        new_name = input("New database name: ").strip()
        if new_name:
            current_db = new_name
            print(f"✓ Switched to '{current_db}'. Place documents in '{current_db}/' and run /build.")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                current_db = names[idx]
                print(f"✓ Switched to '{current_db}'. Documents folder: '{current_db}/'")
            else:
                print("Invalid number.")
        except ValueError:
            current_db = choice
            print(f"✓ Switched to '{current_db}'. Place documents in '{current_db}/' and run /build.")


def run_repl():
    print("=" * 50)
    print("RAG Database Manager")
    print("=" * 50)
    print("Commands: /db              — list / switch / create a database")
    print("          /list            — list indexed documents")
    print("          /build           — scan <db>/ folder and embed new files")
    print("          /add <file>      — embed a specific file")
    print("          /remove <name>   — remove a document by name")
    print("          /quit            — exit\n")

    while True:
        try:
            raw = input(f"[{current_db}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        if raw == "/quit":
            print("Goodbye!")
            break
        elif raw == "/db":
            cmd_db()
        elif raw == "/list":
            cmd_list()
        elif raw == "/build":
            cmd_build()
        elif raw.startswith("/add "):
            file_path = raw[5:].strip()
            cmd_add(file_path)
        elif raw == "/add":
            print("Usage: /add <path>")
        elif raw.startswith("/remove "):
            name = raw[8:].strip()
            cmd_remove(name)
        elif raw == "/remove":
            print("Usage: /remove <name>")
        else:
            print(f"Unknown command: '{raw}'. Type /quit to exit.")


if __name__ == "__main__":
    run_repl()
