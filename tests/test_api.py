import asyncio

import pytest

import api
import immich


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body if json_body is not None else {}
        self.text = text

    def json(self):
        return self._json_body


class _FakeAsyncClient:
    """Records every call so tests can assert on which Immich endpoints were hit."""

    def __init__(self, calls, responses, *a, **kw):
        self.calls = calls
        self.responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def _dispatch(self, method, url, **kw):
        self.calls.append((method, url, kw.get("json")))
        for key, resp in self.responses.items():
            if key in url:
                return resp
        return _FakeResponse(200)

    async def delete(self, url, **kw):
        return await self._dispatch("DELETE", url, **kw)

    async def post(self, url, **kw):
        return await self._dispatch("POST", url, **kw)

    async def put(self, url, **kw):
        return await self._dispatch("PUT", url, **kw)

    async def request(self, method, url, **kw):
        return await self._dispatch(method, url, **kw)


@pytest.fixture(autouse=True)
def reset_review_tag():
    immich.REVIEW_TAG = "Review"
    immich._review_tag_id = "tag-9"
    immich._review_tag_resolved = True
    yield
    immich.REVIEW_TAG = ""
    immich._review_tag_id = None
    immich._review_tag_resolved = False


def test_reset_pet_immich_removes_review_tag(monkeypatch, tmp_path):
    config = {"fido": {"person_id": "person-1"}}
    refs = [{"asset_id": "asset-1", "crop_idx": None, "bbox": None, "face_id": "face-1"}]

    monkeypatch.setattr(api.data, "load_config", lambda data_dir: config)
    monkeypatch.setattr(api.data, "save_config", lambda cfg, data_dir: None)
    monkeypatch.setattr(api.data, "load_pet_refs", lambda person_id, data_dir: refs if person_id == "person-1" else [])
    monkeypatch.setattr(api.data, "save_pet_refs", lambda person_id, refs, data_dir: None)
    monkeypatch.setattr(api, "PETS_DIR", tmp_path)

    calls = []
    responses = {
        "/api/people/person-1": _FakeResponse(200),
        "/api/people": _FakeResponse(201, {"id": "person-2"}),
        "/api/tags/tag-9/assets": _FakeResponse(200, []),
    }
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda *a, **kw: _FakeAsyncClient(calls, responses, *a, **kw))

    asyncio.run(api.reset_pet_immich("fido"))

    untag_calls = [c for c in calls if c[0] == "DELETE" and "/api/tags/tag-9/assets" in c[1]]
    assert untag_calls == [("DELETE", f"{immich.IMMICH_URL}/api/tags/tag-9/assets", {"ids": ["asset-1"]})], (
        "reset_pet_immich must strip the review tag from every ref's asset, same as delete_pet does"
    )
