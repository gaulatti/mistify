from types import SimpleNamespace

import pytest

from src.operations.models import OperationEnvelope
from src.operations.worker import OperationWorker


@pytest.mark.asyncio
async def test_analyze_posts_uses_hash_when_monitor_payload_has_no_id(monkeypatch):
    captured = {}

    async def fake_analysis(request, http_request):
        captured["item"] = request.items[0]
        return SimpleNamespace(results=[])

    monkeypatch.setattr("src.operations.worker.analysis.unified_analysis", fake_analysis)
    app_state = SimpleNamespace(
        config={"PROCESSING_TRANSLATE_TO_ENGLISH": True},
    )
    worker = OperationWorker(SimpleNamespace(), app_state)
    envelope = OperationEnvelope(
        operation_type="analyze_posts",
        payload={
            "items": [
                {
                    "post": {
                        "source": "youtube",
                        "uri": "https://youtube.example/video",
                        "content": "Video title",
                        "createdAt": "2026-08-06T00:00:00Z",
                        "hash": "video-hash",
                    }
                }
            ]
        },
    )

    await worker._analyze_posts(envelope)

    assert captured["item"].id == "video-hash"
