import os

os.environ["AI_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["DATABASE_URL"] = "sqlite:///./test_insightvault.db"
os.environ["CHUNK_SIZE"] = "350"
os.environ["CHUNK_OVERLAP"] = "60"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app


@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def refund_payload():
    return {
        "title": "Refund Policy",
        "content": (
            "Customers can request a refund within 30 days of purchase. "
            "Refunds are not available for final sale items. "
            "Approved refunds return to the original payment method."
        ),
        "source_type": "policy",
        "tags": ["support", "billing"],
    }

