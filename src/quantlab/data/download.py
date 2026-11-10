"""Cached, checksum-verified downloads with a provenance manifest.

A file whose SHA-256 matches its pinned checksum is never downloaded again.
HTTP 401/403/407/451 responses are reported and *not* retried or worked
around (no alternative identities, no mirrors of mirrors).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .sources import RawFile

USER_AGENT = "nba-quant-lab/0.1 (non-commercial sports analytics research)"
DENIED = {401, 403, 407, 451}


class AccessDenied(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, retries: int = 4, timeout: float = 120.0) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(part, dest)
            return
        except urllib.error.HTTPError as e:
            if e.code in DENIED:
                raise AccessDenied(f"{url}: HTTP {e.code} (access refused; not retried)") from e
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** (attempt + 1))


def fetch(files: list[RawFile], raw_dir: Path, log: Callable[[str], None] = print) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for rf in files:
        dest = raw_dir / rf.filename
        if dest.exists() and sha256(dest) == rf.sha256:
            origin = manifest.get(rf.filename, {}).get("origin", "cache")
        else:
            log(f"[fetch] downloading {rf.filename}")
            _download(rf.url, dest)
            digest = sha256(dest)
            if digest != rf.sha256:
                raise RuntimeError(f"{rf.filename}: SHA-256 {digest} != pinned {rf.sha256}")
            origin = "download"
        manifest[rf.filename] = {
            "url": rf.url, "repo": rf.repo, "commit": rf.commit, "sha256": rf.sha256,
            "bytes": dest.stat().st_size, "license": rf.license, "origin": origin,
            "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)
    log(f"[fetch] {len(files)} files verified in {raw_dir}")
    return manifest
