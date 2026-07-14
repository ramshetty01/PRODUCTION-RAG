from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import DEFAULT_DB_PATH
from src.rag.generation import generate_answer
from src.rag.retrieval import DEFAULT_TOP_K, load_vectorstore, retrieve_chunks


def parse_args():
    parser = argparse.ArgumentParser(description="Query the persisted RAG vector database.")
    parser.add_argument("query", help="Question to answer from retrieved chunks.")
    parser.add_argument("--persist-dir", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def main():
    args = parse_args()
    vectorstore = load_vectorstore(args.persist_dir)
    chunks = retrieve_chunks(args.query, vectorstore, top_k=args.top_k)
    response = generate_answer(args.query, chunks)

    print(response["answer"])
    if response["citations"]:
        print("\nCitations:")
        for citation in response["citations"]:
            print(f"- {citation['id']} {citation['source']} page {citation['page']}")


if __name__ == "__main__":
    main()
