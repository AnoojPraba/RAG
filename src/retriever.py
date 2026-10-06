from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL, TOP_K, MIN_SIMILARITY

class DocumentRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """
        Retrieve relevant chunks for a query.
        Returns list of dicts with: content, source, score
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)

        retrieved = []
        for doc, score in results:
            # Score is distance, convert to similarity (0-1)
            similarity = 1 - (score / 2)

            if similarity >= MIN_SIMILARITY:
                retrieved.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": round(similarity, 3)
                })

        return retrieved
