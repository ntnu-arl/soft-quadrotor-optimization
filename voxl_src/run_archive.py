"""Helpers for reading experiment run directories that may be stored as archives.

A "run directory" is either a plain directory or a tar archive (optionally
compressed with zstd/xz/gz/bz2) whose first-level entry is the run directory
itself.  Archives are extracted once into a deterministic directory under
``/tmp`` so later processes can reuse them and callers can treat run sources
uniformly:

    from run_archive import resolve_run_source

    run_dir = resolve_run_source(RUN_ROOT / "2026_07_10_193611_success2_tube_3_camera")

works whether the run is still a directory or has been replaced by
``2026_07_10_193611_success2_tube_3_camera.tar.zst``.
"""

import hashlib
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

ARCHIVE_SUFFIXES = (".tar.zst", ".tar.xz", ".tar.gz", ".tgz", ".tar")

CACHE_ROOT = Path(tempfile.gettempdir()) / "drone_run_archives"
CACHE_MARKER = ".extraction_complete"


def _cache_destination(archive):
    archive = Path(archive).resolve()
    stem = archive.name[: -len(archive_suffix(archive))]
    path_hash = hashlib.sha256(os.fsencode(archive)).hexdigest()[:12]
    return CACHE_ROOT / f"{stem}_{path_hash}"


def archive_suffix(path):
    joined = "".join(Path(path).suffixes)
    return joined if joined in ARCHIVE_SUFFIXES else None


def is_run_archive(path):
    path = Path(path)
    return path.is_file() and archive_suffix(path) is not None


def _extract_archive(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    suffix = archive_suffix(archive)
    if suffix == ".tar.zst":
        # Python < 3.14 tarfile has no native zstd support; stream through the CLI.
        with subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE) as proc:
            try:
                with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                    tar.extractall(dest, filter="data")
            finally:
                proc.stdout.close()
                proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"zstd failed while extracting {archive}")
    else:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(dest, filter="data")


def _extracted_root(dest):
    entries = sorted(entry for entry in dest.iterdir() if entry.name != CACHE_MARKER)
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


def resolve_run_source(source):
    """Return a filesystem directory for a run given a directory or an archive.

    ``source`` may be a directory, an archive file, or a path whose directory
    was replaced by ``<name>.tar.zst`` (the archive sibling is used then).
    Extracted archives are cached under ``/tmp/drone_run_archives`` and reused
    by later processes.
    """
    source = Path(source)
    if source.is_dir():
        return source

    if is_run_archive(source):
        archive = source
    else:
        archive = None
        for suffix in ARCHIVE_SUFFIXES:
            candidate = source.parent / (source.name + suffix)
            if candidate.is_file():
                archive = candidate
                break
        if archive is None:
            raise FileNotFoundError(
                f"No run directory or run archive found for {source}"
            )

    dest = _cache_destination(archive)
    marker = dest / CACHE_MARKER
    if dest.exists() and not marker.is_file():
        raise RuntimeError(
            f"Incomplete archive cache at {dest}; remove it and run again"
        )
    if not marker.is_file():
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}_", dir=CACHE_ROOT))
        try:
            _extract_archive(archive, staging)
            (staging / CACHE_MARKER).touch()
            staging.rename(dest)
        except BaseException:
            if staging.exists():
                import shutil
                shutil.rmtree(staging)
            raise
    return _extracted_root(dest)


def resolve_run_children(root):
    """Yield usable run directories for every child of ``root``.

    Children that are plain directories are returned as-is; children that are
    archives are extracted (cached) first.  Non-directory, non-archive entries
    (e.g. loose archives that are not run folders are still returned; callers
    filter by name).
    """
    root = Path(root)
    for child in sorted(root.iterdir()):
        if child.is_dir() or is_run_archive(child):
            yield child, resolve_run_source(child)
