import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import chromadb
import os
from tqdm import tqdm
import readline
import atexit
from pathlib import Path
import glob

DB_FOLDER = "vector_db"
EMBEDDING_MODEL = "nomic-embed-text"

# Active collection name — changed at runtime by /db
current_db = "docs"

def docs_folder():
    """Each DB keeps its documents in a folder with the same name."""
    return current_db


# --- Document loading ---

_converter = None

def _get_converter():
    """Lazy-load the docling DocumentConverter (loads ML models once)."""
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            generate_page_images=False,      # don't keep rendered page bitmaps in memory
            generate_picture_images=False,   # don't keep figure bitmaps in memory
        )
        _converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )
    return _converter

def _stitch_parts(parts):
    """Join batch-converted markdown, stitching tables and headings that were
    split at page-batch boundaries so that chunk_text() sees coherent blocks.

    When a table continues across a batch boundary the new batch has only data
    rows — the header and separator are missing.  We re-inject them so that
    every table section is self-contained and parseable.
    """
    import re
    if not parts:
        return ""

    table_row = re.compile(r'^\s*\|')
    sep_re    = re.compile(r'^\s*\|(?:[-:\s]+\|)+\s*$')
    result = parts[0]

    for nxt in parts[1:]:
        tail = [l for l in result.rstrip().split('\n') if l.strip()]
        head = [l for l in nxt.lstrip().split('\n') if l.strip()]
        last  = tail[-1] if tail else ''
        first = head[0]  if head else ''

        # Table continues across the boundary
        if table_row.match(last) and table_row.match(first):
            # Does nxt already have its own header+separator?
            # A proper table starts with a non-separator row followed by a separator.
            second_is_sep = len(head) >= 2 and sep_re.match(head[1])
            first_is_sep  = sep_re.match(first)

            if not first_is_sep and not second_is_sep:
                # Continuation rows without a header — re-inject from result
                header = _extract_table_header(result)
                if header:
                    nxt = header + '\n' + nxt.lstrip()

            result = result.rstrip() + '\n' + nxt.lstrip()

        # Orphaned heading at end of batch → attach body from next batch
        elif last.startswith('#') and not first.startswith('#'):
            result = result.rstrip() + '\n' + nxt.lstrip()
        else:
            result = result + '\n\n' + nxt

    return result


def _batch_worker(path, start, end, output_file):
    """Subprocess worker: converts pages *start*–*end-1* of *path* to markdown
    and writes the result to *output_file*.  Running in a dedicated subprocess
    ensures that all ML-model memory (layout / table-structure networks) is
    fully reclaimed by the OS when this process exits — it never leaks back
    into the parent."""
    import io
    import pypdf
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.document import DocumentStream

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
    )
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )

    with open(path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        writer = pypdf.PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        buf = io.BytesIO()
        writer.write(buf)

    buf.seek(0)
    result = converter.convert(DocumentStream(name="batch.pdf", stream=buf))
    markdown = result.document.export_to_markdown()

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(markdown)


def _build_enriched_chunk(chunk):
    """Build an enriched chunk by prepending metadata to the content.
    Format:
    Document: <source>
    
    Hierarchy:
    <h1>
    <h2>
    <h3>
    
    <chunk_content>
    """
    lines = []
    
    if chunk.metadata.get("source"):
        lines.append(f"Document: {chunk.metadata['source']}")
        lines.append("")
    
    # Build hierarchy section
    hierarchy = []
    if chunk.metadata.get("h1"):
        hierarchy.append(chunk.metadata.get("h1"))
    if chunk.metadata.get("h2"):
        hierarchy.append(chunk.metadata.get("h2"))
    if chunk.metadata.get("h3"):
        hierarchy.append(chunk.metadata.get("h3"))
    
    if hierarchy:
        lines.append("Hierarchy:")
        for item in hierarchy:
            lines.append(item)
        lines.append("")
    lines.append("Content:")
    lines.append(chunk.page_content)
    return "\n".join(lines)


def load_pdf(path):
    """Convert PDF to markdown via docling, BATCH_SIZE pages at a time.

    Each batch runs in a *subprocess* (spawn context).  When the subprocess
    exits the OS reclaims all of its memory — including the several GB of
    ML-model weights that docling loads for layout/table detection.  This is
    the only reliable way to prevent RAM+swap exhaustion on large PDFs: Python
    GC and manual del/gc.collect() do not return heap memory to the OS.
    """
    import gc
    import pypdf
    import tempfile
    import multiprocessing

    BATCH_SIZE = 5

    with open(path, 'rb') as f:
        total_pages = len(pypdf.PdfReader(f).pages)

    ranges = [(s, min(s + BATCH_SIZE, total_pages))
              for s in range(0, total_pages, BATCH_SIZE)]

    stitched = ""

    for (s, e) in tqdm(ranges, desc="  Converting PDF", unit="batch",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches  pp{postfix}  [{elapsed}<{remaining}]",
                       leave=True):
        tqdm.write(f"    pages {s + 1}–{e} of {total_pages}")

        # Write markdown to a temp file — avoids serialising large strings over IPC
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md")
        os.close(tmp_fd)

        try:
            # 'spawn' starts a clean interpreter with no inherited model state
            ctx = multiprocessing.get_context('spawn')
            proc = ctx.Process(target=_batch_worker, args=(path, s, e, tmp_path))
            proc.start()
            proc.join()

            if proc.exitcode != 0:
                raise RuntimeError(
                    f"Batch conversion failed (exit code {proc.exitcode}) "
                    f"for pages {s + 1}–{e}"
                )

            with open(tmp_path, 'r', encoding='utf-8') as f:
                part = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Stitch incrementally — parent only ever holds one batch + accumulated text
        stitched = _stitch_parts([stitched, part]) if stitched else part
        del part
        gc.collect()

    return stitched

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_document(path):
    if path.lower().endswith('.pdf'):
        return load_pdf(path), True   # (text, is_markdown)
    is_md = path.lower().endswith('.md')
    return load_text(path), is_md

def chunk_text(text, markdown=False, debug=False, source_name=None):
    """Split text into chunks. Uses header-aware splitting for markdown output.
    Tables are kept intact and merged into the adjacent chunk (appended to the
    last preceding chunk, or prepended to the next chunk if none exists yet).
    Tracks heading hierarchy: when a higher-level heading appears, updates all
    parent levels. Chunks inherit the full heading path including all ancestors.
    Returns list of (chunk_text, metadata_dict) tuples for markdown, plain strings for plain text."""
    import re

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", " "],
        chunk_size=1500,
        chunk_overlap=300
    )
    if not markdown:
        # Plain text: return strings (will be wrapped in embed_file)
        return splitter.split_text(text)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False
    )

    # FIRST: Apply header splitting to the entire document to preserve heading hierarchy
    header_docs = header_splitter.split_text(text)
    
    # Match a table block: 2+ consecutive lines that start with '|'
    table_pattern = re.compile(r'((?:^|\n)(?:\|[^\n]+\n){2,}\|[^\n]+)', re.MULTILINE)

    #all_chunks = []
    '''
    # Track active headings as we process documents in order
    active_headings = {"h1": None, "h2": None, "h3": None}

    for doc_idx, doc in enumerate(header_docs):
        # Update active headings based on this document's metadata
        # Each document tells us what heading section it's in
        if "h1" in doc.metadata:
            new_h1 = doc.metadata["h1"]
            if new_h1 != active_headings["h1"]:
                active_headings["h1"] = new_h1
                active_headings["h2"] = None
                active_headings["h3"] = None
        
        if "h2" in doc.metadata:
            new_h2 = doc.metadata["h2"]
            if new_h2 != active_headings["h2"]:
                active_headings["h2"] = new_h2
                active_headings["h3"] = None
        
        if "h3" in doc.metadata:
            new_h3 = doc.metadata["h3"]
            if new_h3 != active_headings["h3"]:
                active_headings["h3"] = new_h3
        
        if debug:
            # Only show when there's new heading metadata
            has_heading = any(doc.metadata.get(k) for k in ["h1", "h2", "h3"])
            if has_heading:
                print(f"  Doc {doc_idx}: metadata={doc.metadata}, active_headings now={active_headings}")
        
        # Build heading metadata dict with level info and document name
        
        heading_metadata = {
            "h1": active_headings["h1"],
            "h2": active_headings["h2"],
            "h3": active_headings["h3"]
        }
        
        # Add document information if provided
        if source_name:
            heading_metadata["source"] = source_name
            heading_metadata["document"] = os.path.splitext(source_name)[0]
        
        # NOW: Split the document content by tables
        # re.split with a capturing group yields [text, table, text, table, ...]
        parts = table_pattern.split(doc.page_content)
        
        pending_tables = []  # tables waiting to be prepended to next chunk
        text_chunks = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Text segment — chunk further with RecursiveCharacterTextSplitter
                for chunk_content in splitter.split_text(part):
                    if chunk_content.strip():
                        text_chunks.append((chunk_content, heading_metadata.copy()))
                
                # Flush pending tables onto first text chunk
                if text_chunks and pending_tables:
                    content, hmeta = text_chunks[0]
                    text_chunks[0] = ("\n\n".join(pending_tables) + "\n\n" + content, hmeta)
                    pending_tables = []
            else:
                # Table segment — attach to last chunk or queue for next
                table = part.strip()
                if not table:
                    continue
                if text_chunks:
                    content, hmeta = text_chunks[-1]
                    text_chunks[-1] = (content + "\n\n" + table, hmeta)
                else:
                    pending_tables.append(table)

        # Tables at end of this header section
        if pending_tables:
            if text_chunks:
                content, hmeta = text_chunks[-1]
                text_chunks[-1] = (content + "\n\n" + "\n\n".join(pending_tables), hmeta)
            else:
                # Table with no text under this heading — carry forward to next header section
                for table in pending_tables:
                    all_chunks.append((table, heading_metadata.copy()))
        
        all_chunks.extend(text_chunks)
    '''
    chunks = header_splitter.split_text(text)
    for chunk in chunks:
        chunk.metadata["source"] = source_name
        print(chunk.metadata)
        print(chunk.page_content)
    #return header_splitter.split_text(text)  # Return header-split chunks without table handling for now
    return chunks


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
    print(f"  [{source_name}] loading...")
    text, is_markdown = load_document(file_path)

    print(f"  [{source_name}] chunking...")
    chunks = chunk_text(text, markdown=is_markdown, source_name=source_name)
    print(f"  [{source_name}] {len(chunks)} chunks — embedding...")

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks),
                              desc="  Embedding", unit="chunk",
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]",
                              leave=False):
        
        # Build enriched chunk content with metadata prefix
        enriched_content = _build_enriched_chunk(chunk)
        
        # Filter out None values for ChromaDB (it doesn't accept None in metadatas)
        #clean_metadata = {k: v for k, v in metadata.items() if v is not None}
        
        emb = ollama.embed(model=EMBEDDING_MODEL, input=enriched_content).embeddings[0]
        collection.add(
            ids=[f"{source_name}__{i}"],
            embeddings=[emb],
            documents=[chunk.page_content],  # Store plain chunk text; enriched version used only for embedding
            metadatas=[chunk.metadata]
        )
    return len(chunks)


# --- Subcommands ---

def cmd_convert(file_path):
    """Convert a single PDF or text file to a .md file in the same directory."""
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return
    if not file_path.lower().endswith(('.pdf', '.txt')):
        print(f"Only .pdf or .txt files can be converted.")
        return

    dst_path = os.path.splitext(file_path)[0] + ".md"
    if os.path.exists(dst_path):
        answer = input(f"'{dst_path}' already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print(f"Converting '{os.path.basename(file_path)}'...")
    try:
        text, _ = load_document(file_path)
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✓ Saved → {dst_path}")
        print("  Run /build to embed it into the vector database.")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")

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

def cmd_add(file_name, debug=False):
    folder = docs_folder()
    file_path = os.path.join(folder, file_name)
    
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return
    if not file_name.lower().endswith('.md'):
        print(f"Only markdown (.md) files can be embedded.")
        return
    
    collection = open_collection(create=True)
    if file_name in get_embedded_sources(collection):
        print(f"'{file_name}' is already in the database. Remove it first to re-embed.")
        return
    print(f"Embedding '{file_name}'...")
    
    if debug:
        # Show what will be embedded
        print(f"\n{'='*60}")
        print(f"DEBUG: Showing enriched chunks that will be embedded")
        print(f"{'='*60}\n")
        
        text, is_markdown = load_document(file_path)
        chunks = chunk_text(text, markdown=is_markdown, source_name=file_name)
        
        for i, chunk_data in enumerate(chunks[:3], 1):  # Show first 3
            if is_markdown and isinstance(chunk_data, tuple):
                chunk_content, chunk_metadata = chunk_data
                enriched = _build_enriched_chunk(chunk_content, chunk_metadata)
                clean_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}
                
                print(f"--- Chunk {i}/{len(chunks)} (will be embedded) ---")
                print(enriched)
                print(f"\nMetadata stored in ChromaDB: {clean_metadata}")
                print()
        
        if len(chunks) > 3:
            print(f"... ({len(chunks) - 3} more chunks)")
        print(f"{'='*60}\n")
    
    count = embed_file(collection, file_path, file_name)
    print(f"✓ Added {count} chunks from '{file_name}'.")

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
        print(f"Created '{folder}/' — add files and run /convert, then /build.")
        return
    files = [f for f in os.listdir(folder) if f.endswith('.md')]
    if not files:
        print(f"No markdown files found in '{folder}/'. Run /convert first.")
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
    for file in tqdm(new_files, desc="Files", unit="file"):
        file_path = os.path.join(folder, file)
        try:
            count = embed_file(collection, file_path, file)
            tqdm.write(f"  ✓ {file}: {count} chunks")
        except Exception as e:
            tqdm.write(f"  ✗ {file}: {e}")
    print(f"\n✓ Done! {collection.count()} total embeddings, {len(files)} documents.")


def cmd_chunk(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return
    print(f"Loading '{os.path.basename(file_path)}'...")
    text, is_markdown = load_document(file_path)
    source_name = os.path.basename(file_path)
    chunks = chunk_text(text, markdown=is_markdown, debug=False, source_name=source_name)
    print(f"\n{'='*50}")
    print(f"Chunks: {len(chunks)}  |  Splitter: {'markdown-aware' if is_markdown else 'plain text'}")
    print(f"{'='*50}\n")
    for i, chunk_data in enumerate(chunks, 1):
        print(f"--- Chunk {i}/{len(chunks)} ({len(chunk_data.page_content)} chars) ---")
        print(chunk_data)
        print()
        if i < len(chunks):
            answer = input("[Enter] next  [q] quit  [a] all > ").strip().lower()
            if answer == "q":
                break
            elif answer == "a":
                for j, remaining_data in enumerate(chunks[i:], i + 1):
                    print(f"--- Chunk {j}/{len(chunks)} ({len(remaining_data.page_content)} chars) ---")
                    print(remaining_data)
                    print()
                break
    print(f"✓ {len(chunks)} chunks total from '{os.path.basename(file_path)}'.")


def cmd_normalize(file_path):
    """Normalize heading levels in a markdown file.
    For numbered headings (e.g. '1 Title', '1.2 Sub', '1.2.3 Sub-sub'),
    the depth is derived from the number of components (dots + 1).
    For unnumbered headings, the level is kept relative to their position.
    """
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return
    if not file_path.lower().endswith('.md'):
        print(f"Only markdown (.md) files can be normalized.")
        return
    
    print(f"Analyzing '{os.path.basename(file_path)}'...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    import re
    heading_pattern = re.compile(r'^(#+)\s+(.+)$')
    # Matches numbered headings: "1 Title", "1.2 Title", "1.2.3 Title", etc.
    numbered_heading = re.compile(r'^(\d+(?:\.\d+)*)\s+\S')
    
    normalized_lines = []
    changes = 0
    
    for line in lines:
        match = heading_pattern.match(line)
        if match:
            old_hashes = match.group(1)
            heading_text = match.group(2).rstrip()
            old_level = len(old_hashes)
            
            num_match = numbered_heading.match(heading_text)
            if num_match:
                # Depth = number of dot-separated components
                new_level = len(num_match.group(1).split('.'))
            else:
                # Unnumbered heading: keep its existing level
                new_level = old_level
            
            new_level = min(new_level, 6)
            new_hashes = '#' * new_level
            new_line = f"{new_hashes} {heading_text}\n"
            
            if new_line != line:
                changes += 1
                print(f"  {old_hashes} {heading_text[:60]}  →  {new_hashes}")
            
            normalized_lines.append(new_line)
        else:
            normalized_lines.append(line)
    
    if changes == 0:
        print(f"✓ No changes needed - headings already normalized.")
        return
    
    answer = input(f"\nApply {changes} heading changes? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(normalized_lines)
    
    print(f"✓ Normalized {changes} headings in '{os.path.basename(file_path)}'.")


def cmd_delete():
    global current_db
    client = get_client()
    try:
        collection = open_collection()
        count = collection.count()
    except Exception:
        print(f"Database '{current_db}' does not exist.")
        return

    answer = input(
        f"Delete database '{current_db}' ({count} embeddings)? This cannot be undone. [y/N] "
    ).strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    client.delete_collection(current_db)
    print(f"✓ Deleted database '{current_db}'.")
    current_db = "docs"
    print(f"Switched back to default database 'docs'.")


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
    print(f"  d. Delete current database")

    choice = input("\nSelect number or type name to create: ").strip()
    if not choice:
        return
    if choice.lower() == "d":
        cmd_delete()
        return
    elif choice.lower() == "n":
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
    # Setup readline history (up arrow to access previous commands)
    history_file = Path.home() / ".rag_history"
    if history_file.exists():
        readline.read_history_file(history_file)
    readline.set_history_length(10)
    atexit.register(readline.write_history_file, history_file)
    
    print("=" * 50)
    print("RAG Database Manager")
    print("=" * 50)
    print("Commands: /db              — list / switch / create a database")
    print("          /db delete       — delete the current database")
    print("          /list            — list indexed documents")
    print("          /convert <path>  — convert PDF/txt files to .md (supports *.pdf patterns)")
    print("          /normalize <path>— normalize heading levels in markdown file")
    print("          /build           — embed .md files from <db>/")
    print("          /add <file>      — embed a specific .md file")
    print("          /add <file> debug — embed and show what will be stored")
    print("          /remove <name>   — remove a document by name")
    print("          /chunk <file>    — preview how a file will be chunked")
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
        elif raw in ("/db", "/db delete"):
            if raw == "/db delete":
                cmd_delete()
            else:
                cmd_db()
        elif raw == "/list":
            cmd_list()
        elif raw.startswith("/convert "):
            pattern = raw[9:].strip()
            # Expand glob patterns
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"No files match: {pattern}")
            else:
                print(f"\nFound {len(files)} file(s) to convert:")
                for i, f in enumerate(files, 1):
                    print(f"  {i}. {f}")
                confirm = input(f"\nProceed with conversion? [y/N] ").strip().lower()
                if confirm == "y":
                    for file_path in files:
                        cmd_convert(file_path)
                else:
                    print("Aborted.")
        elif raw == "/convert":
            print("Usage: /convert <path>  (e.g., /convert docs/*.pdf or /convert docs/file.pdf)")
        elif raw.startswith("/normalize "):
            file_path = raw[11:].strip()
            cmd_normalize(file_path)
        elif raw == "/normalize":
            print("Usage: /normalize <path>  (e.g., /normalize docs/file.md)")
        elif raw == "/build":
            cmd_build()
        elif raw.startswith("/add "):
            args = raw[5:].strip()
            # Check if debug flag is at the end
            debug = args.endswith(" debug")
            if debug:
                file_path = args[:-6].strip()  # Remove " debug" suffix
            else:
                file_path = args
            
            if not file_path:
                print("Usage: /add <file> [debug]")
                return
            
            cmd_add(file_path, debug=debug)
        elif raw == "/add":
            print("Usage: /add <file> [debug]")
        elif raw.startswith("/remove "):
            name = raw[8:].strip()
            cmd_remove(name)
        elif raw == "/remove":
            print("Usage: /remove <name>")
        elif raw.startswith("/chunk "):
            file_path = raw[7:].strip()
            cmd_chunk(file_path)
        elif raw == "/chunk":
            print("Usage: /chunk <path>")
        else:
            print(f"Unknown command: '{raw}'. Type /quit to exit.")


if __name__ == "__main__":
    run_repl()
