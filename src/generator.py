import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.config import MODEL, MAX_TOKENS, TEMPERATURE

class AnswerGenerator:
    def __init__(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.llm = ChatOpenAI(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

    def generate(self, query: str, retrieved_docs: list[dict]) -> dict:
        """
        Generate an answer using Claude, citing retrieved sources.
        """
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information in the indexed documents.",
                "citations": []
            }

        # Format context from retrieved documents
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
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions based on provided documents.

Use the retrieved documents below to answer the question. Always cite sources using [1], [2], etc.
If you use information from a document, include its citation.

Retrieved Documents:
{context}

Question: {question}

Answer:"""
        )

        # Generate answer
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": query
        })

        return {
            "answer": response.content,
            "citations": citations
        }
