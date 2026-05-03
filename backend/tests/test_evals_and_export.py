def test_eval_runner_returns_scores(client):
    response = client.post("/api/evals/run")

    assert response.status_code == 200
    data = response.json()
    assert data["eval_count"] >= 4
    assert data["results"][0]["average_score"] >= 0
    assert "retrieval_hit" in data["results"][0]["question_results"][0]


def test_markdown_summary_export(client, refund_payload):
    created = client.post("/api/documents", json=refund_payload).json()

    response = client.get(f"/api/documents/{created['document_id']}/summary.md")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# Refund Policy" in response.text
    assert "## Summary" in response.text


def test_app_import_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

