from app.services.chunking_service import ChunkingService


def test_document_ingestion_creates_chunks(client, refund_payload):
    response = client.post("/api/documents", json=refund_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == 1
    assert data["ingestion_status"] == "completed"
    assert data["chunk_count"] >= 1

    history = client.get("/api/documents").json()
    assert history[0]["title"] == "Refund Policy"
    assert history[0]["chunk_count"] >= 1


def test_chunking_service_tracks_offsets():
    content = "A" * 120 + ". " + "B" * 120 + ". " + "C" * 120
    chunks = ChunkingService(chunk_size=140, chunk_overlap=20).chunk_text(content)

    assert len(chunks) >= 2
    assert chunks[0].character_start == 0
    assert chunks[0].character_end > chunks[0].character_start
    assert chunks[1].character_start < chunks[0].character_end


def test_async_job_status_flow(client, refund_payload):
    response = client.post("/api/documents/async", json=refund_payload)

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = client.get(f"/api/jobs/{job_id}")

    assert status.status_code == 200
    assert status.json()["status"] in {"queued", "processing", "completed"}
    if status.json()["status"] == "completed":
        assert status.json()["document_id"] is not None

