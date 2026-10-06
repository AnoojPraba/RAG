import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import DOCUMENTS_DIR, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

class DocumentIndexer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )

    def ingest_file(self, file_path: str) -> dict:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load document
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        elif file_path.suffix.lower() in [".txt", ".md", ".c", ".cpp", ".h", ".hpp"]:
            loader = TextLoader(str(file_path))
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Set metadata
        for doc in documents:
            doc.metadata["source"] = file_path.name

        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Add to vector store
        ids = self.vector_store.add_documents(chunks)
        self.vector_store.persist()

        return {
            "file": file_path.name,
            "chunks": len(chunks),
            "ids": ids
        }

    def get_all_documents(self) -> list:
        """Get metadata about all indexed documents."""
        collection = self.vector_store._collection
        if not collection or not hasattr(collection, "get"):
            return []

        all_data = collection.get()
        sources = set()
        if all_data and "metadatas" in all_data:
            for metadata in all_data["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])

        return sorted(list(sources))

    def delete_document(self, filename: str) -> bool:
        """Delete all chunks from a specific document."""
        collection = self.vector_store._collection
        if not collection:
            return False

        all_data = collection.get()
        if not all_data or "metadatas" not in all_data:
            return False

        ids_to_delete = []
        for i, metadata in enumerate(all_data["metadatas"]):
            if metadata and metadata.get("source") == filename:
                ids_to_delete.append(all_data["ids"][i])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            self.vector_store.persist()
            return True

        return False

    def ingest_folder(self, folder_path: str) -> dict:
        """Ingest all PDFs and text files from a folder."""
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        supported_extensions = {".pdf", ".txt", ".md", ".c", ".cpp", ".h", ".hpp"}
        files = [
            f for f in folder_path.rglob("*")
            if f.suffix.lower() in supported_extensions and f.is_file()
        ]

        if not files:
            return {"folder": str(folder_path), "files": 0, "total_chunks": 0}

        results = {
            "folder": str(folder_path),
            "files": 0,
            "total_chunks": 0,
            "details": []
        }

        for file_path in files:
            try:
                result = self.ingest_file(str(file_path))
                results["files"] += 1
                results["total_chunks"] += result["chunks"]
                results["details"].append({
                    "file": result["file"],
                    "chunks": result["chunks"]
                })
            except Exception as e:
                results["details"].append({
                    "file": file_path.name,
                    "error": str(e)
                })

        return results
