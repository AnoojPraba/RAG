import requests
from src.config import MAX_TOKENS, TEMPERATURE

class OllamaGenerator:
    def __init__(self, model: str = "mistral"):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.check_connection()

    def check_connection(self):
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            raise ConnectionError(
                f"Ollama not running. Start with:\n"
                f"  ollama serve\n"
                f"Then in another terminal:\n"
                f"  ollama pull {self.model}"
            )

    def generate(self, query: str, retrieved_docs: list[dict]) -> dict:
        """Generate answer using local Ollama model."""
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information in the indexed documents.",
                "citations": []
            }

        # Format context
        context_parts = []
        citations = []

        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"[{i+1}] {doc['content']}\nSource: {doc['source']}")
            citations.append({
                "id": i + 1,
                "source": doc["source"],
                "score": doc["score"]
            })

        context = "\n\n".join(context_parts)

        # Create prompt
        prompt = f"""You are a helpful assistant answering questions based on provided documents.

Use the retrieved documents below to answer the question. Always cite sources using [1], [2], etc.
If you use information from a document, include its citation.

Retrieved Documents:
{context}

Question: {query}

Answer:"""

        # Generate with Ollama
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": TEMPERATURE,
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()
            answer = response.json()["response"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

        return {
            "answer": answer.strip(),
            "citations": citations,
            "model": self.model
        }

    def list_models(self) -> list[str]:
        """List available models on Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except requests.exceptions.RequestException:
            return []
