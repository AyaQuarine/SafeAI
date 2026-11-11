from fastapi.testclient import TestClient

from safeai.ui.app import create_app


def test_health_and_demo_endpoints() -> None:
    client = TestClient(create_app())
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["sample_rate"] == 16000

    page = client.get("/")
    assert page.status_code == 200
    assert "Acquisition audio" in page.text

    demo = client.post("/api/inspect/demo")
    assert demo.status_code == 200
    payload = demo.json()
    assert payload["chunks_generated"] > 0
    assert payload["waveform"]["times"]
    assert all("ndarray" not in str(event) for event in payload["quality_events"])
    assert payload["timeline"]


def test_microphone_start_requires_consent() -> None:
    client = TestClient(create_app())
    response = client.post("/api/session/start", data={"consent": "false"})
    assert response.status_code == 400
