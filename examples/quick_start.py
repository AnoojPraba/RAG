#!/usr/bin/env python
"""Quick start example for the RAG system."""

import os
from dotenv import load_dotenv
from src.indexer import DocumentIndexer
from src.retriever import DocumentRetriever
from src.generator import AnswerGenerator

load_dotenv()

def main():
    # Initialize components
    indexer = DocumentIndexer()
    retriever = DocumentRetriever()
    generator = AnswerGenerator()

    # 1. Index a document
    print("1️⃣  Indexing sample notes...")
    sample_file = "examples/sample_notes.txt"
    if os.path.exists(sample_file):
        result = indexer.ingest_file(sample_file)
        print(f"   ✓ Indexed {result['chunks']} chunks from {result['file']}\n")
    else:
        print(f"   ℹ️  Sample file not found. Create one at {sample_file}\n")

    # 2. List indexed documents
    print("2️⃣  Listing indexed documents...")
    docs = indexer.get_all_documents()
    if docs:
        for doc in docs:
            print(f"   - {doc}")
    else:
        print("   (No documents indexed yet)")
    print()

    # 3. Ask a question
    questions = [
        "What is supervised learning?",
        "Explain the difference between overfitting and underfitting",
        "What metrics should I use to evaluate my model?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"3.{i}️⃣  Question: {question}")

        # Retrieve relevant chunks
        retrieved = retriever.retrieve(question, k=3)

        if retrieved:
            # Generate answer
            response = generator.generate(question, retrieved)

            print(f"\n    Answer:\n    {response['answer']}\n")
            print(f"    Citations:")
            for citation in response['citations']:
                print(f"      [{citation['id']}] {citation['source']} (score: {citation['score']})")
        else:
            print("    (No relevant documents found)\n")

        print()

if __name__ == "__main__":
    main()
