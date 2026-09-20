"""Download the MIND dataset from its official (gated) Hugging Face mirror.

The MIND download buttons on https://msnews.github.io/ point at
https://huggingface.co/datasets/yjw1029/MIND, which is gated behind the Microsoft
Research license. Accepting the gate on that page and supplying a read token via
``NEWSREC_HF_TOKEN`` is the licensed path; there is no anonymous download.

Raw archives and their extracted contents stay under ``data/`` and are never committed.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from news_recsys.config import EXPECTED_ZIP_BYTES, HF_BASE_URL, Settings, get_settings
from news_recsys.logging_utils import get_logger

logger = get_logger("data.download")

#: Files we expect inside each MIND split archive.
SPLIT_FILES = ("behaviors.tsv", "news.tsv")

_CHUNK = 1 << 20


class DatasetAccessError(RuntimeError):
    """Raised when the gated dataset cannot be reached with the configured token."""


@dataclass(frozen=True)
class DownloadResult:
    split: str
    zip_path: Path
    extract_dir: Path
    bytes_downloaded: int
    cached: bool


def _common_prefix(members: list[str]) -> str:
    """Return the single shared top-level directory of a zip, if there is one.

    MIND archives nest everything under e.g. ``MINDsmall_train/``; we strip that so the
    layout on disk is ``data/raw/<variant>/<split>/behaviors.tsv`` regardless of variant.
    """
    tops = {member.split("/", 1)[0] for member in members if "/" in member}
    if len(tops) == 1 and all("/" in member for member in members):
        return f"{tops.pop()}/"
    return ""


def _auth_headers(settings: Settings) -> dict[str, str]:
    if not settings.hf_token:
        raise DatasetAccessError(
            "NEWSREC_HF_TOKEN is not set. Accept the license at "
            "https://huggingface.co/datasets/yjw1029/MIND and put a read token in .env"
        )
    return {"Authorization": f"Bearer {settings.hf_token}"}


def download_split(
    split: str, settings: Settings | None = None, *, force: bool = False
) -> DownloadResult:
    """Download and extract one MIND split. Idempotent: re-runs are no-ops."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    zip_name = settings.zip_names[split]
    zip_path = settings.raw_dir / zip_name
    extract_dir = settings.raw_dir / split
    expected = EXPECTED_ZIP_BYTES.get(zip_name)

    already_extracted = all((extract_dir / name).exists() for name in SPLIT_FILES)
    if already_extracted and not force:
        logger.info("%s already extracted at %s", split, extract_dir)
        return DownloadResult(
            split, zip_path, extract_dir, zip_path.stat().st_size if zip_path.exists() else 0, True
        )

    if not zip_path.exists() or force or (expected and zip_path.stat().st_size != expected):
        url = f"{HF_BASE_URL}/{zip_name}"
        logger.info("downloading %s -> %s", url, zip_path)
        with requests.get(
            url, headers=_auth_headers(settings), stream=True, timeout=120
        ) as response:
            if response.status_code in (401, 403):
                raise DatasetAccessError(
                    f"HTTP {response.status_code} for {zip_name}: the token cannot read the gated "
                    "dataset. Accept the license at https://huggingface.co/datasets/yjw1029/MIND."
                )
            response.raise_for_status()
            tmp_path = zip_path.with_suffix(".zip.part")
            written = 0
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_CHUNK):
                    handle.write(chunk)
                    written += len(chunk)
            tmp_path.replace(zip_path)
        logger.info("downloaded %s (%.1f MiB)", zip_name, written / 1024 / 1024)

    size = zip_path.stat().st_size
    if expected is not None and size != expected:
        raise DatasetAccessError(
            f"{zip_name} is {size} bytes, expected {expected}. Delete it and retry."
        )

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if not m.endswith("/")]
        prefix = _common_prefix(members)
        for member in members:
            relative = member[len(prefix) :] if prefix else member
            target = extract_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as sink:
                while chunk := source.read(_CHUNK):
                    sink.write(chunk)
    logger.info("extracted %d files to %s", len(members), extract_dir)

    missing = [name for name in SPLIT_FILES if not (extract_dir / name).exists()]
    if missing:
        raise DatasetAccessError(f"{zip_name} is missing expected files: {missing}")

    return DownloadResult(split, zip_path, extract_dir, size, False)


def download_all(settings: Settings | None = None, *, force: bool = False) -> list[DownloadResult]:
    settings = settings or get_settings()
    return [download_split(split, settings, force=force) for split in settings.zip_names]
