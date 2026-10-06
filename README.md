# RAG: Chat with Your Notes

A Python CLI tool to index personal notes and PDFs, then ask questions with cited sources.

## Features
- **Ingest** PDFs, text files, and code (C/C++)
- **Search** with semantic similarity
- **Answer** questions with Claude, citing sources
- **Manage** indexed documents

## Supported File Types
- `.txt`, `.md` — text/markdown
- `.pdf` — PDF documents
- `.c`, `.cpp`, `.h`, `.hpp` — C/C++ source code

## Quick Start

### Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option A: Use Ollama (Local, Free, No API Key)
```bash
# Install Ollama: https://ollama.ai

# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull a model
ollama pull mistral    # ~4GB, fast & good quality
# or
ollama pull llama2     # ~3.8GB, good for code

# List available models
python rag.py ollama-models

# Ask questions with --local flag
python rag.py ask "What is this about?" --local
python rag.py ask "Explain the function" --local --model llama2
```

### Option B: Use Claude API (Faster, Better Quality)
```bash
export ANTHROPIC_API_KEY="sk-..."
python rag.py ask "Your question?"
```

### Ingest documents
```bash
# Index a single file
python rag.py ingest path/to/notes.txt
python rag.py ingest path/to/document.pdf
python rag.py ingest path/to/code.cpp

# Index all supported files in a folder (recursive)
python rag.py ingest-folder path/to/notes/folder
```

### Ask questions
```bash
# With Claude API (if ANTHROPIC_API_KEY is set)
python rag.py ask "What are the main points about topic X?"

# With local Ollama (no API key needed)
python rag.py ask "What are the main points about topic X?" --local

# Specify which Ollama model
python rag.py ask "question" --local --model llama2
```

### List indexed documents
```bash
python rag.py list
```

## How It Works

1. **Index** → PDF/text split into 512-char chunks, embedded with sentence-transformers (local), stored in ChromaDB
2. **Query** → Question embedded (local), search for top-k similar chunks (local)
3. **Generate** → Send chunks to Claude API OR local Ollama, get answer
4. **Cite** → Responses include source document and chunk reference

## Retrieval is 100% Local
- Embeddings: sentence-transformers (runs on your machine)
- Vector DB: ChromaDB (stored on disk)
- Search: Pure local, no external API calls

Only the **answer generation** step uses an external API (Claude) or local LLM (Ollama).

## Ollama Models (Free, Local)

Popular models optimized for RAG:

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| mistral | 4GB | Fast | Good | General Q&A |
| llama2 | 3.8GB | Medium | Good | Code & docs |
| neural-chat | 4GB | Fast | Good | Conversational |
| orca-mini | 2.7GB | Very Fast | Fair | Quick answers |

Download: `ollama pull mistral`

## Project Structure
```
RAG/
├── rag.py              # CLI entry point
├── src/
│   ├── indexer.py      # Document ingestion
│   ├── retriever.py     # Vector search (local)
│   ├── generator.py     # Claude API generation
│   ├── ollama_generator.py  # Local LLM generation
│   └── config.py       # Settings
├── data/documents/     # Source files
└── chroma_db/          # Vector store (auto-created)
```
