#!/usr/bin/env python
import os
import click
from dotenv import load_dotenv
from src.indexer import DocumentIndexer
from src.retriever import DocumentRetriever
from src.generator import AnswerGenerator
from src.ollama_generator import OllamaGenerator

load_dotenv()

@click.group()
def cli():
    """RAG: Chat with your indexed notes and documents."""
    pass

@cli.command()
@click.argument("file_path")
def ingest(file_path):
    """Index a PDF or text file."""
    try:
        indexer = DocumentIndexer()
        result = indexer.ingest_file(file_path)
        click.echo(
            f"✓ Indexed {result['file']}: {result['chunks']} chunks added"
        )
    except FileNotFoundError as e:
        click.echo(f"✗ Error: {e}", err=True)
    except ValueError as e:
        click.echo(f"✗ Error: {e}", err=True)

@cli.command()
@click.argument("folder_path")
def ingest_folder(folder_path):
    """Index all PDFs and text files in a folder (recursive)."""
    try:
        indexer = DocumentIndexer()
        result = indexer.ingest_folder(folder_path)

        if result["files"] == 0:
            click.echo(f"ℹ️  No documents found in {result['folder']}")
            return

        click.echo(f"✓ Indexed {result['files']} file(s): {result['total_chunks']} total chunks")
        for detail in result["details"]:
            if "error" in detail:
                click.echo(f"  ⚠️  {detail['file']}: {detail['error']}")
            else:
                click.echo(f"  ✓ {detail['file']}: {detail['chunks']} chunks")

    except ValueError as e:
        click.echo(f"✗ Error: {e}", err=True)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)

@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of sources to retrieve")
@click.option("--local", is_flag=True, help="Use local Ollama instead of Claude API")
@click.option("--model", default="mistral", help="Ollama model to use (with --local)")
def ask(query, top_k, local, model):
    """Ask a question about your documents."""
    try:
        retriever = DocumentRetriever()
        click.echo("Searching documents...")
        docs = retriever.retrieve(query, k=top_k)

        if not docs:
            click.echo("No relevant documents found.")
            return

        if local:
            click.echo(f"Generating answer with local {model}...")
            generator = OllamaGenerator(model=model)
        elif os.getenv("ANTHROPIC_API_KEY"):
            click.echo("Generating answer with Claude...")
            generator = AnswerGenerator()
        else:
            click.echo("\nRETRIEVED DOCUMENTS (no API key or --local flag)\n")
            click.echo("="*60)
            for i, doc in enumerate(docs, 1):
                click.echo(f"\n[{i}] {doc['source']} (similarity: {doc['score']})")
                click.echo(f"    {doc['content'][:200]}...")
            click.echo("\n" + "="*60)
            click.echo("Tip: Use --local to answer with Ollama (no API key needed)")
            return

        result = generator.generate(query, docs)

        click.echo("\n" + "="*60)
        click.echo("ANSWER")
        click.echo("="*60)
        click.echo(result["answer"])

        click.echo("\n" + "="*60)
        click.echo("CITATIONS")
        click.echo("="*60)
        for citation in result["citations"]:
            click.echo(
                f"[{citation['id']}] {citation['source']} "
                f"(similarity: {citation['score']})"
            )

        if local:
            click.echo(f"\nGenerated with: {model}")

    except ConnectionError as e:
        click.echo(f"Connection Error: {e}\n", err=True)
        click.echo("Make sure Ollama is running:", err=True)
        click.echo("  ollama serve", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
def list():
    """List all indexed documents."""
    try:
        indexer = DocumentIndexer()
        docs = indexer.get_all_documents()

        if not docs:
            click.echo("No documents indexed yet.")
            return

        click.echo("\n📚 Indexed Documents:")
        for i, doc in enumerate(docs, 1):
            click.echo(f"  {i}. {doc}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)

@cli.command()
@click.argument("filename")
def delete(filename):
    """Delete an indexed document."""
    try:
        indexer = DocumentIndexer()
        if indexer.delete_document(filename):
            click.echo(f"✓ Deleted {filename}")
        else:
            click.echo(f"✗ Document not found: {filename}")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)

@cli.command()
def ollama_models():
    """List available Ollama models."""
    try:
        generator = OllamaGenerator()
        models = generator.list_models()

        if not models:
            click.echo("No models found. Pull one first:")
            click.echo("  ollama pull mistral")
            click.echo("  ollama pull llama2")
            return

        click.echo("\nAvailable Ollama models:")
        for model in models:
            click.echo(f"  - {model}")

        click.echo("\nUse with: python rag.py ask 'question' --local --model <name>")

    except ConnectionError as e:
        click.echo(f"Error: {e}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

if __name__ == "__main__":
    cli()
