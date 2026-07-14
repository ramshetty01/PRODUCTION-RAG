from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_PDF_PATH = Path(__file__).with_name("docs.pdf")
DEFAULT_DB_PATH = Path(__file__).with_name("chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_pdf(file_path=DEFAULT_PDF_PATH, chunk_size=2000, chunk_overlap=400):
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(docs)


def build_chroma_db(chunks, persist_directory=DEFAULT_DB_PATH):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )


if __name__ == "__main__":
    chunks = chunk_pdf()
    print(f"Created {len(chunks)} chunks from {DEFAULT_PDF_PATH.name}")

    vectorstore = build_chroma_db(chunks)
    print(f"Saved Chroma database to {DEFAULT_DB_PATH}")

    if chunks:
        first = chunks[0]
        source = first.metadata.get("source", DEFAULT_PDF_PATH.name)
        page = first.metadata.get("page", "unknown")
        preview = first.page_content[:300].replace("\n", " ")
        print(f"First chunk: source {source}, page {page}")
        print(preview)
