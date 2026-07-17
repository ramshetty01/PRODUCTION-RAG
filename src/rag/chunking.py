import csv
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_PATH = PROJECT_ROOT / "docs.pdf"
DEFAULT_DB_PATH = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_TOKENS = 700
DEFAULT_CHUNK_OVERLAP_TOKENS = 100
MIN_PDF_PAGE_TEXT_TOKENS = 8

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm", ".csv", ".docx", ".pptx"}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def count_tokens(text):
    return len(tokenize(text))


def chunk_token_summary(chunks) -> dict:
    token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
    return {
        "chunks": len(token_counts),
        "min_tokens": min(token_counts) if token_counts else None,
        "max_tokens": max(token_counts) if token_counts else None,
        "target_tokens": DEFAULT_CHUNK_TOKENS,
        "overlap_tokens": DEFAULT_CHUNK_OVERLAP_TOKENS,
    }


def load_pdf(file_path=DEFAULT_PDF_PATH):
    file_path = Path(file_path).expanduser().resolve()
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    return _apply_pdf_ocr_fallback(file_path, docs)


def is_low_text_pdf_page(doc: Document, min_tokens: int = MIN_PDF_PAGE_TEXT_TOKENS) -> bool:
    return count_tokens(doc.page_content) < min_tokens


def _apply_pdf_ocr_fallback(file_path: Path, docs: list[Document]) -> list[Document]:
    low_text_pages = [doc for doc in docs if is_low_text_pdf_page(doc)]
    if not low_text_pages:
        return docs

    ocr_pages = ocr_pdf_pages(file_path, [int(doc.metadata.get("page") or 0) for doc in low_text_pages])
    if not ocr_pages:
        return docs

    by_page = {int(page): text for page, text in ocr_pages.items() if text.strip()}
    output = []
    for doc in docs:
        page = int(doc.metadata.get("page") or 0)
        text = by_page.get(page)
        if text:
            output.append(
                Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "source": str(file_path),
                        "page": page,
                        "parser": "pdf-ocr",
                        "ocr": True,
                    },
                )
            )
        else:
            output.append(doc)
    return output


def ocr_pdf_pages(file_path: str | Path, pages: list[int]) -> dict[int, str]:
    if not pages:
        return {}
    try:
        import pdf2image
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR fallback requires optional local dependencies: install pytesseract, pdf2image, "
            "Tesseract OCR, and Poppler."
        ) from exc

    file_path = Path(file_path).expanduser().resolve()
    extracted = {}
    for page in pages:
        images = pdf2image.convert_from_path(str(file_path), first_page=page + 1, last_page=page + 1)
        text = "\n".join(pytesseract.image_to_string(image).strip() for image in images)
        if text.strip():
            extracted[page] = text
    return extracted


def _document_metadata(file_path: Path, page: int = 0, section: str | None = None) -> dict:
    metadata = {
        "source": str(file_path),
        "page": page,
        "document_id": file_path.stem,
        "parser": file_path.suffix.lower().lstrip(".") or "text",
    }
    if section:
        metadata["section"] = section
    return metadata


def load_plain_text_document(file_path: str | Path):
    file_path = Path(file_path).expanduser().resolve()
    return [Document(page_content=file_path.read_text(encoding="utf-8"), metadata=_document_metadata(file_path))]


def load_html_document(file_path: str | Path):
    file_path = Path(file_path).expanduser().resolve()
    parser = TextExtractor()
    parser.feed(file_path.read_text(encoding="utf-8"))
    return [Document(page_content=parser.text(), metadata=_document_metadata(file_path))]


def load_csv_document(file_path: str | Path):
    file_path = Path(file_path).expanduser().resolve()
    docs = []
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            text = " | ".join(cell.strip() for cell in row if cell.strip())
            if text:
                docs.append(Document(page_content=text, metadata=_document_metadata(file_path, page=index, section="row")))
    return docs


def _zip_xml_text(file_path: Path, members: list[str]) -> list[str]:
    texts = []
    try:
        with zipfile.ZipFile(file_path) as archive:
            for member in members:
                try:
                    root = ElementTree.fromstring(archive.read(member))
                except KeyError:
                    continue
                parts = [node.text.strip() for node in root.iter() if node.text and node.text.strip()]
                if parts:
                    texts.append(" ".join(parts))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid {file_path.suffix.lower()} file: {file_path.name}") from exc
    return texts


def load_docx_document(file_path: str | Path):
    file_path = Path(file_path).expanduser().resolve()
    texts = _zip_xml_text(file_path, ["word/document.xml"])
    return [
        Document(page_content=text, metadata=_document_metadata(file_path, page=index, section="body"))
        for index, text in enumerate(texts)
    ]


def load_pptx_document(file_path: str | Path):
    file_path = Path(file_path).expanduser().resolve()
    try:
        with zipfile.ZipFile(file_path) as archive:
            members = sorted(member for member in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", member))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid {file_path.suffix.lower()} file: {file_path.name}") from exc
    texts = _zip_xml_text(file_path, members)
    return [
        Document(page_content=text, metadata=_document_metadata(file_path, page=index, section=f"slide-{index + 1}"))
        for index, text in enumerate(texts)
    ]


def load_document(file_path: str | Path):
    file_path = Path(file_path).expanduser().resolve()
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    if suffix in {".md", ".markdown", ".txt"}:
        return load_plain_text_document(file_path)
    if suffix in {".html", ".htm"}:
        return load_html_document(file_path)
    if suffix == ".csv":
        return load_csv_document(file_path)
    if suffix == ".docx":
        return load_docx_document(file_path)
    if suffix == ".pptx":
        return load_pptx_document(file_path)
    raise ValueError(f"unsupported file type: {suffix or '<none>'}")


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
    document_version: str = "v1",
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
        document_id = str(doc.metadata.get("document_id") or Path(source).stem or "document")
        section = doc.metadata.get("section")
        access_roles = doc.metadata.get("access_roles", ["public"])

        for chunk_text in split_text(doc.page_content):
            chunk_index = len(chunks)
            chunk_metadata = {
                **doc.metadata,
                "source": source_path,
                "page": page,
                "section": section,
                "document_id": document_id,
                "document_version": str(doc.metadata.get("document_version") or document_version),
                "access_roles": access_roles,
                "chunk_index": chunk_index,
                "chunk_id": f"{document_id}:p{page}:c{chunk_index}",
            }
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

    return chunks


def chunk_pdf(
    file_path=DEFAULT_PDF_PATH,
    chunk_size=DEFAULT_CHUNK_TOKENS,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS,
    document_version: str = "v1",
):
    docs = load_pdf(file_path)
    return chunk_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        document_version=document_version,
    )


def load_text_document(file_path):
    return load_plain_text_document(file_path)


def chunk_text_file(
    file_path,
    chunk_size=DEFAULT_CHUNK_TOKENS,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS,
    document_version: str = "v1",
):
    docs = load_text_document(file_path)
    return chunk_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        document_version=document_version,
    )


def chunk_file(
    file_path,
    chunk_size=DEFAULT_CHUNK_TOKENS,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS,
    document_version: str = "v1",
):
    docs = load_document(file_path)
    return chunk_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        document_version=document_version,
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
