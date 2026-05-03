from app.schemas import Citation, RetrievedChunk
from app.services.guardrail_service import GuardrailService
from app.services.llm_provider import get_llm_provider


def test_retrieval_returns_relevant_chunks(client, refund_payload):
    client.post("/api/documents", json=refund_payload)

    response = client.post("/api/retrieve", json={"query": "What is the refund window?", "top_k": 3})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert results[0]["document_title"] == "Refund Policy"
    assert "30 days" in results[0]["text"]


def test_ask_endpoint_returns_grounded_answer_with_citations(client, refund_payload):
    client.post("/api/documents", json=refund_payload)

    response = client.post(
        "/api/ask",
        json={"question": "What is the refund window?", "top_k": 3, "answer_style": "concise"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "30 days" in data["answer"]
    assert data["citations"]
    assert data["guardrail_result"]["passed"] is True
    assert data["model_metadata"]["provider"] == "mock"


def test_ask_endpoint_refuses_when_no_context_exists(client):
    response = client.post("/api/ask", json={"question": "What is the refund window?", "top_k": 3})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I could not find enough information in the uploaded documents."
    assert data["citations"] == []
    assert data["guardrail_result"]["passed"] is False


def test_insight_extraction_returns_valid_schema(client):
    response = client.post(
        "/api/documents",
        json={
            "title": "Checkout Migration Notes",
            "content": (
                "Decision: migrate checkout validation in phase two. "
                "Action: Priya must prepare rollback steps by Friday. "
                "Risk: gateway certification may delay launch."
            ),
            "source_type": "meeting_notes",
        },
    )
    document_id = response.json()["document_id"]

    insight_response = client.post("/api/insights", json={"document_id": document_id})

    assert insight_response.status_code == 200
    data = insight_response.json()
    assert data["summary"]
    assert data["action_items"]
    assert data["guardrail_result"]["passed"] is True


def test_mock_provider_is_default_without_api_key():
    assert get_llm_provider().provider_name == "mock"


def test_guardrail_catches_missing_citation():
    guardrail = GuardrailService().validate_answer(
        answer="Refunds are allowed within 30 days.",
        citations=[],
        retrieved_context=[
            RetrievedChunk(
                chunk_id="doc-1-chunk-1",
                document_title="Refund Policy",
                score=0.9,
                text="Customers can request a refund within 30 days.",
                citation="[Refund Policy, chunk 1]",
            )
        ],
    )

    assert guardrail.passed is False
    assert "missing citations" in " ".join(guardrail.errors)


def test_guardrail_deduplicates_citations():
    citations = [
        Citation(citation="[Refund Policy, chunk 1]", document_title="Refund Policy", chunk_id="doc-1-chunk-1"),
        Citation(citation="[Refund Policy, chunk 1]", document_title="Refund Policy", chunk_id="doc-1-chunk-1"),
    ]

    assert len(GuardrailService().dedupe_citations(citations)) == 1

