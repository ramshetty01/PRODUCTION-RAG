import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_PATH = PROJECT_ROOT / "docs.pdf"
DEFAULT_DB_PATH = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_TOKENS = 700
DEFAULT_CHUNK_OVERLAP_TOKENS = 100

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def count_tokens(text):
    return len(tokenize(text))


def load_pdf(file_path=DEFAULT_PDF_PATH):
    file_path = Path(file_path).expanduser().resolve()
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def create_token_splitter(
    chunk_size=DEFAULT_CHUNK_TOKENS,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_text(text):
        tokens = tokenize(text)
        if not tokens:
            return []

        step = chunk_size - chunk_overlap
        chunks = []
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start : start + chunk_size]
            if not chunk_tokens:
                break
            chunks.append(" ".join(chunk_tokens))
            if start + chunk_size >= len(tokens):
                break
        return chunks

    return split_text


def chunk_documents(
    docs,
    chunk_size=DEFAULT_CHUNK_TOKENS,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS,
):
    split_text = create_token_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []

    for doc in docs:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page")
        source_path = str(Path(source).expanduser().resolve()) if source else ""

        for chunk_text in split_text(doc.page_content):
            chunk_index = len(chunks)
            chunk_metadata = {
                **doc.metadata,
                "source": source_path,
                "page": page,
                "chunk_index": chunk_index,
                "chunk_id": f"{Path(source).stem or 'document'}:p{page}:c{chunk_index}",
            }
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

    return chunks


def chunk_pdf(
    file_path=DEFAULT_PDF_PATH,
    chunk_size=DEFAULT_CHUNK_TOKENS,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS,
):
    docs = load_pdf(file_path)
    return chunk_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


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
    token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
    if token_counts:
        print(
            "Token counts: "
            f"min={min(token_counts)}, max={max(token_counts)}, "
            f"target={DEFAULT_CHUNK_TOKENS}, overlap={DEFAULT_CHUNK_OVERLAP_TOKENS}"
        )

    if chunks:
        first = chunks[0]
        source = first.metadata.get("source", DEFAULT_PDF_PATH.name)
        page = first.metadata.get("page", "unknown")
        chunk_index = first.metadata.get("chunk_index", "unknown")
        preview = first.page_content[:300].replace("\n", " ")
        print(f"First chunk: source {source}, page {page}, chunk {chunk_index}")
        print(preview)
