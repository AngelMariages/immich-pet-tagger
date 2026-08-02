"""Immich HTTP helpers. Sync functions are used by the poller (runs in a thread).
Async functions are used by the API routes."""

import asyncio
import logging
import os
import threading

import httpx
import requests

log = logging.getLogger("immich")

IMMICH_URL = os.environ.get("IMMICH_URL", "http://immich-server:2283").rstrip("/")
IMMICH_API_KEY = os.environ.get("IMMICH_API_KEY", "")

# When set, this Immich tag is applied to an asset every time a face is written to it,
# so tagged photos can be reviewed in Immich. Empty means the feature is off.
REVIEW_TAG = os.environ.get("PET_REVIEW_TAG", "").strip()

FACE_BOX_SIZE = 256

_owner_id: str | None = None


def headers() -> dict:
    return {"x-api-key": IMMICH_API_KEY, "Accept": "application/json"}


def get_owner_id() -> str | None:
    """Return the user ID of the API key owner, cached after first call."""
    global _owner_id
    if _owner_id is None:
        try:
            r = requests.get(f"{IMMICH_URL}/api/users/me", headers=headers(), timeout=10)
            if r.status_code == 200:
                _owner_id = r.json().get("id")
        except Exception as e:
            log.warning(f"get_owner_id failed: {e}")
    return _owner_id


# ---------------------------------------------------------------------------
# Review tag
#
# The tag id is resolved once (creating the tag if it does not exist) and cached
# for the process lifetime. Applying it is a single PUT per asset issued inline
# with the face creation, so a face and its review tag always land together --
# nothing is batched or deferred to a later pass.
# ---------------------------------------------------------------------------

_review_tag_id: str | None = None
_review_tag_lock = threading.Lock()
_review_tag_alock = asyncio.Lock()


def _tag_id_from_upsert(payload) -> str | None:
    """Pick our tag out of a TagResponseDto list, matching on the full tag path."""
    if not isinstance(payload, list):
        return None
    for tag in payload:
        if not isinstance(tag, dict):
            continue
        if tag.get("value") == REVIEW_TAG or tag.get("name") == REVIEW_TAG:
            return tag.get("id")
    if len(payload) == 1 and isinstance(payload[0], dict):
        return payload[0].get("id")
    return None


def resolve_review_tag_id_sync() -> str | None:
    """Return the id of REVIEW_TAG, creating the tag on first use. None if disabled/failed."""
    global _review_tag_id
    if not REVIEW_TAG:
        return None
    if _review_tag_id:
        return _review_tag_id
    with _review_tag_lock:
        if _review_tag_id:
            return _review_tag_id
        try:
            r = requests.put(
                f"{IMMICH_URL}/api/tags",
                json={"tags": [REVIEW_TAG]},
                headers={**headers(), "Content-Type": "application/json"},
                timeout=15,
            )
            if r.status_code in (200, 201):
                _review_tag_id = _tag_id_from_upsert(r.json())
            else:
                log.warning(f"review tag upsert -> {r.status_code}: {r.text[:200]}")
        except Exception as e:
            log.warning(f"review tag upsert failed: {e}")
        if not _review_tag_id:
            log.warning(f"Could not resolve review tag '{REVIEW_TAG}'; assets will not be tagged.")
        else:
            log.info(f"Review tag '{REVIEW_TAG}' -> {_review_tag_id}")
        return _review_tag_id


async def resolve_review_tag_id(client: httpx.AsyncClient) -> str | None:
    """Async twin of resolve_review_tag_id_sync, sharing the same cached id."""
    global _review_tag_id
    if not REVIEW_TAG:
        return None
    if _review_tag_id:
        return _review_tag_id
    async with _review_tag_alock:
        if _review_tag_id:
            return _review_tag_id
        try:
            resp = await client.put(
                f"{IMMICH_URL}/api/tags",
                json={"tags": [REVIEW_TAG]},
                headers={**headers(), "Content-Type": "application/json"},
                timeout=15,
            )
            if resp.status_code in (200, 201):
                _review_tag_id = _tag_id_from_upsert(resp.json())
            else:
                log.warning(f"review tag upsert -> {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            log.warning(f"review tag upsert failed: {e}")
        if not _review_tag_id:
            log.warning(f"Could not resolve review tag '{REVIEW_TAG}'; assets will not be tagged.")
        else:
            log.info(f"Review tag '{REVIEW_TAG}' -> {_review_tag_id}")
        return _review_tag_id


def _log_tag_result(asset_id: str, payload) -> None:
    """Immich answers with a per-id result list; 'duplicate' just means already tagged."""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and not item.get("success") and item.get("error") != "duplicate":
                log.warning(f"review tag on {asset_id}: {item.get('error')}")


def apply_review_tag_sync(asset_id: str) -> None:
    """Tag asset_id with REVIEW_TAG (no-op when unset). Never raises: a tagging
    failure must not undo or fail the face that was just created."""
    tag_id = resolve_review_tag_id_sync()
    if not tag_id:
        return
    try:
        r = requests.put(
            f"{IMMICH_URL}/api/tags/{tag_id}/assets",
            json={"ids": [asset_id]},
            headers={**headers(), "Content-Type": "application/json"},
            timeout=15,
        )
        if r.status_code != 200:
            log.warning(f"review tag on {asset_id} -> {r.status_code}: {r.text[:200]}")
            return
        _log_tag_result(asset_id, r.json())
    except Exception as e:
        log.warning(f"review tag on {asset_id} failed: {e}")


async def apply_review_tag(client: httpx.AsyncClient, asset_id: str) -> None:
    """Async twin of apply_review_tag_sync."""
    tag_id = await resolve_review_tag_id(client)
    if not tag_id:
        return
    try:
        resp = await client.put(
            f"{IMMICH_URL}/api/tags/{tag_id}/assets",
            json={"ids": [asset_id]},
            headers={**headers(), "Content-Type": "application/json"},
            timeout=15,
        )
        if resp.status_code != 200:
            log.warning(f"review tag on {asset_id} -> {resp.status_code}: {resp.text[:200]}")
            return
        _log_tag_result(asset_id, resp.json())
    except Exception as e:
        log.warning(f"review tag on {asset_id} failed: {e}")


# ---------------------------------------------------------------------------
# Sync (poller)
# ---------------------------------------------------------------------------

def fetch_assets_created_after(created_after_iso: str) -> list[tuple[str, str]]:
    """Return [(asset_id, createdAt_iso), ...] for the background poller.
    Uses createdAt (upload time) so photos synced late never fall behind the cutoff."""
    return _fetch_assets({"createdAfter": created_after_iso}, ts_field="createdAt", label="fetch_assets_created_after")


def fetch_assets_taken_after(taken_after_iso: str, taken_before_iso: str | None = None) -> list[tuple[str, str]]:
    """Return [(asset_id, fileCreatedAt_iso), ...] for manual scans.
    Uses takenAfter (EXIF date) so the date picker matches what the user sees in the Immich library."""
    query: dict = {"takenAfter": taken_after_iso}
    if taken_before_iso:
        query["takenBefore"] = taken_before_iso
    return _fetch_assets(query, ts_field="fileCreatedAt", label="fetch_assets_taken_after")


def _fetch_assets(query: dict, ts_field: str, label: str) -> list[tuple[str, str]]:
    url = f"{IMMICH_URL}/api/search/metadata"
    hdrs = {**headers(), "Content-Type": "application/json"}
    out: list[tuple[str, str]] = []
    page = 1
    size = 1000
    while True:
        r = requests.post(url, json={**query, "page": page, "size": size, "order": "asc"}, headers=hdrs, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"{label}: HTTP {r.status_code} on page {page}: {r.text[:200]}")
        data = r.json()
        block = data.get("assets") or {}
        items = (block.get("items") if isinstance(block, dict) else None) or data.get("items") or []
        owner_id = get_owner_id()
        for a in items:
            aid = a.get("id")
            ts = a.get(ts_field) or a.get("localDateTime") or ""
            if aid and ts:
                if owner_id and a.get("ownerId") != owner_id:
                    continue
                out.append((str(aid).strip("\x00"), ts))
        if len(items) < size:
            break
        page += 1
    return out


def fetch_face_id_for_person(asset_id: str, person_id: str) -> str | None:
    """Return the face_id on asset_id that belongs to person_id, or None."""
    try:
        r = requests.get(f"{IMMICH_URL}/api/faces", params={"id": asset_id}, headers=headers(), timeout=10)
        if r.status_code == 200:
            for face in r.json():
                if (face.get("person") or {}).get("id") == person_id:
                    return face.get("id")
    except Exception as e:
        log.warning(f"fetch_face_id_for_person {asset_id}: {e}")
    return None


def fetch_asset_face_person_ids(asset_id: str) -> set[str]:
    """Return set of person_ids already assigned as faces on this asset."""
    try:
        r = requests.get(f"{IMMICH_URL}/api/faces", params={"id": asset_id}, headers=headers(), timeout=10)
        if r.status_code != 200 or not isinstance(r.json(), list):
            return set()
        return {str(f["person"]["id"]) for f in r.json() if (f.get("person") or {}).get("id")}
    except Exception:
        return set()


def post_face_sync(asset_id: str, person_id: str, bbox_norm=None, img_size=None) -> str | None:
    """Create a face entry in Immich (sync, used by poller). Returns face_id on success, None on failure."""
    if bbox_norm is not None and img_size is not None:
        x1, y1, x2, y2 = bbox_norm
        iw, ih = img_size
        bx, by = int(x1 * iw), int(y1 * ih)
        bw, bh = int((x2 - x1) * iw), int((y2 - y1) * ih)
    else:
        bx, by, bw, bh = 0, 0, FACE_BOX_SIZE, FACE_BOX_SIZE
        iw, ih = FACE_BOX_SIZE, FACE_BOX_SIZE
    try:
        r = requests.post(
            f"{IMMICH_URL}/api/faces",
            json={"assetId": asset_id, "personId": person_id,
                  "width": bw, "height": bh,
                  "imageWidth": iw, "imageHeight": ih,
                  "x": bx, "y": by},
            headers={**headers(), "Content-Type": "application/json"},
            timeout=30,
        )
        if r.status_code not in (200, 201):
            log.warning(f"post_face {asset_id} -> {r.status_code}: {r.text[:200]}")
            return None
        apply_review_tag_sync(asset_id)
        fr = requests.get(f"{IMMICH_URL}/api/faces", headers=headers(), params={"id": asset_id}, timeout=15)
        if fr.status_code == 200:
            for face in fr.json():
                if (face.get("person") or {}).get("id") == person_id:
                    return face.get("id")
        log.warning(f"post_face: created but could not retrieve face_id for asset {asset_id}")
        return None
    except Exception as e:
        log.error(f"post_face error: {e}")
        return None


# ---------------------------------------------------------------------------
# Async (API routes)
# ---------------------------------------------------------------------------

async def post_face(client: httpx.AsyncClient, asset_id: str, person_id: str, bbox_norm=None) -> str | None:
    """Create a face entry in Immich. Returns face_id on success, None on failure.
    Immich returns 201 with empty body, so face_id is fetched via GET after creation.
    Immich scales the box by the supplied imageWidth/imageHeight, so a normalized
    bbox can be mapped onto a fixed virtual canvas without fetching the image."""
    if bbox_norm is not None:
        x1, y1, x2, y2 = bbox_norm
        s = FACE_BOX_SIZE
        bx, by = int(x1 * s), int(y1 * s)
        bw, bh = int((x2 - x1) * s), int((y2 - y1) * s)
    else:
        bx, by, bw, bh = 0, 0, FACE_BOX_SIZE, FACE_BOX_SIZE
    try:
        resp = await client.post(
            f"{IMMICH_URL}/api/faces",
            headers={**headers(), "Content-Type": "application/json"},
            json={"assetId": asset_id, "personId": person_id,
                  "width": bw, "height": bh,
                  "imageWidth": FACE_BOX_SIZE, "imageHeight": FACE_BOX_SIZE,
                  "x": bx, "y": by},
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            log.warning(f"post_face failed {resp.status_code}: {resp.text[:200]}")
            return None
        await apply_review_tag(client, asset_id)
        faces_resp = await client.get(f"{IMMICH_URL}/api/faces", headers=headers(), params={"id": asset_id})
        if faces_resp.status_code == 200:
            for face in faces_resp.json():
                if (face.get("person") or {}).get("id") == person_id:
                    return face.get("id")
        log.warning(f"post_face: created but could not retrieve face_id for asset {asset_id}")
        return None
    except Exception as e:
        log.error(f"post_face error: {e}")
        return None


async def get_existing_face_person_ids(client: httpx.AsyncClient, asset_id: str) -> set[str]:
    """Return set of person_ids already assigned as faces on this asset (async)."""
    try:
        resp = await client.get(f"{IMMICH_URL}/api/faces", headers=headers(), params={"id": asset_id}, timeout=15)
        if resp.status_code == 200:
            return {f.get("person", {}).get("id") for f in resp.json() if f.get("person")}
    except Exception as e:
        log.warning(f"get_existing_face_person_ids error: {e}")
    return set()
