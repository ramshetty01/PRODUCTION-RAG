from langchain_core.documents import Document

from src.rag.runtime_quality import score_runtime_answer


def make_doc(text="A runner executes jobs.", chunk_id="docs:p0:c0"):
    return Document(page_content=text, metadata={"chunk_id": chunk_id})


def test_runtime_quality_scores_supported_cited_answers():
    quality = score_runtime_answer("A runner executes jobs. [docs:p0:c0]", [make_doc()])

    assert quality.passed is True
    assert quality.citation_coverage == 1.0
    assert quality.evidence_support == 1.0


def test_runtime_quality_warns_on_unsupported_answer():
    quality = score_runtime_answer("A runner approves payroll vendors. [docs:p0:c0]", [make_doc()])

    assert quality.passed is False
    assert quality.status == "warning"
    assert "answer terms are weakly supported by cited evidence" in quality.reasons
