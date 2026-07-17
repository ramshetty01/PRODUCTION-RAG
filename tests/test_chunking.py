import zipfile

from langchain_core.documents import Document

from src.rag.chunking import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    SUPPORTED_DOCUMENT_SUFFIXES,
    chunk_file,
    chunk_documents,
    chunk_text_file,
    chunk_token_summary,
    count_tokens,
    is_low_text_pdf_page,
    load_pdf,
    ocr_pdf_pages,
)


def test_chunk_documents_uses_token_target_and_preserves_metadata():
    text = " ".join(f"token{i}" for i in range(2200))
    docs = [Document(page_content=text, metadata={"source": "docs.pdf", "page": 0})]

    chunks = chunk_documents(docs)

    assert len(chunks) > 1
    assert all(count_tokens(chunk.page_content) <= DEFAULT_CHUNK_TOKENS for chunk in chunks)
    assert chunks[0].metadata["source"].endswith("docs.pdf")
    assert chunks[0].metadata["page"] == 0
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_id"].startswith("docs:p0:c0")
    assert chunks[0].metadata["document_id"] == "docs"
    assert chunks[0].metadata["document_version"] == "v1"
    assert chunks[0].metadata["access_roles"] == ["public"]
    assert 500 <= DEFAULT_CHUNK_TOKENS <= 800
    assert DEFAULT_CHUNK_OVERLAP_TOKENS == 100


def test_neighboring_chunks_keep_overlap():
    text = " ".join(f"word{i}" for i in range(1200))
    docs = [Document(page_content=text, metadata={"source": "docs.pdf", "page": 2})]

    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS)

    first_tokens = chunks[0].page_content.split()
    second_tokens = chunks[1].page_content.split()

    assert first_tokens[-DEFAULT_CHUNK_OVERLAP_TOKENS:] == second_tokens[
        :DEFAULT_CHUNK_OVERLAP_TOKENS
    ]


def test_chunk_token_summary_reports_verification_counts():
    text = " ".join(f"token{i}" for i in range(900))
    docs = [Document(page_content=text, metadata={"source": "docs.pdf", "page": 0})]

    chunks = chunk_documents(docs)
    summary = chunk_token_summary(chunks)

    assert summary["chunks"] == len(chunks)
    assert summary["max_tokens"] <= DEFAULT_CHUNK_TOKENS
    assert summary["target_tokens"] == DEFAULT_CHUNK_TOKENS
    assert summary["overlap_tokens"] == DEFAULT_CHUNK_OVERLAP_TOKENS


def test_chunk_text_file_ingests_markdown_with_document_metadata(tmp_path):
    source = tmp_path / "security-policy.md"
    source.write_text("# Security Policy\n\nAll vendors require SOC 2 evidence.", encoding="utf-8")

    chunks = chunk_text_file(source, chunk_size=12, chunk_overlap=2, document_version="v3")

    assert chunks
    assert chunks[0].metadata["source"] == str(source.resolve())
    assert chunks[0].metadata["page"] == 0
    assert chunks[0].metadata["document_id"] == "security-policy"
    assert chunks[0].metadata["document_version"] == "v3"
    assert chunks[0].metadata["access_roles"] == ["public"]


def test_chunk_file_supports_html_csv_docx_and_pptx(tmp_path):
    html = tmp_path / "policy.html"
    html.write_text("<html><body><h1>Vendor Policy</h1><p>SOC 2 required.</p></body></html>", encoding="utf-8")
    csv_file = tmp_path / "controls.csv"
    csv_file.write_text("control,evidence\nvendor,SOC 2\nincident,review", encoding="utf-8")
    docx = tmp_path / "brief.docx"
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr(
            "word/document.xml",
            "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
            "<w:body><w:p><w:r><w:t>DOCX vendor evidence</w:t></w:r></w:p></w:body></w:document>",
        )
    pptx = tmp_path / "deck.pptx"
    with zipfile.ZipFile(pptx, "w") as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            "<p:sld xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' "
            "xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'>"
            "<p:cSld><p:spTree><a:t>PPTX audit packet</a:t></p:spTree></p:cSld></p:sld>",
        )

    html_chunks = chunk_file(html, chunk_size=20, chunk_overlap=2)
    csv_chunks = chunk_file(csv_file, chunk_size=20, chunk_overlap=2)
    docx_chunks = chunk_file(docx, chunk_size=20, chunk_overlap=2)
    pptx_chunks = chunk_file(pptx, chunk_size=20, chunk_overlap=2)

    assert {".pdf", ".docx", ".pptx", ".html", ".csv", ".txt", ".md", ".markdown"} <= SUPPORTED_DOCUMENT_SUFFIXES
    assert "Vendor Policy" in html_chunks[0].page_content
    assert csv_chunks[0].metadata["section"] == "row"
    assert any("SOC 2" in chunk.page_content for chunk in csv_chunks)
    assert "DOCX vendor evidence" in docx_chunks[0].page_content
    assert docx_chunks[0].metadata["parser"] == "docx"
    assert "PPTX audit packet" in pptx_chunks[0].page_content
    assert pptx_chunks[0].metadata["section"] == "slide-1"


def test_chunk_file_rejects_unsupported_type(tmp_path):
    source = tmp_path / "archive.json"
    source.write_text("{}", encoding="utf-8")

    try:
        chunk_file(source)
    except ValueError as exc:
        assert "unsupported file type" in str(exc)
    else:
        raise AssertionError("expected unsupported file type to fail")


def test_chunk_file_reports_invalid_office_package(tmp_path):
    source = tmp_path / "broken.docx"
    source.write_text("not a zip", encoding="utf-8")

    try:
        chunk_file(source)
    except ValueError as exc:
        assert "invalid .docx file" in str(exc)
    else:
        raise AssertionError("expected invalid office package to fail")


def test_low_text_pdf_pages_use_ocr_fallback(monkeypatch, tmp_path):
    import src.rag.chunking as chunking

    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF-1.4")
    docs = [Document(page_content="", metadata={"source": str(source), "page": 0})]
    monkeypatch.setattr(chunking.PyPDFLoader, "load", lambda self: docs)
    monkeypatch.setattr(chunking, "ocr_pdf_pages", lambda path, pages: {0: "OCR extracted vendor policy"})

    loaded = load_pdf(source)

    assert is_low_text_pdf_page(docs[0]) is True
    assert loaded[0].page_content == "OCR extracted vendor policy"
    assert loaded[0].metadata["page"] == 0
    assert loaded[0].metadata["parser"] == "pdf-ocr"
    assert loaded[0].metadata["ocr"] is True


def test_ocr_pdf_pages_reports_missing_optional_dependencies(monkeypatch, tmp_path):
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF-1.4")

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in {"pdf2image", "pytesseract"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        ocr_pdf_pages(source, [0])
    except RuntimeError as exc:
        assert "OCR fallback requires optional local dependencies" in str(exc)
    else:
        raise AssertionError("expected missing OCR dependencies to fail clearly")
