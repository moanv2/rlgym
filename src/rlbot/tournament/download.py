"""Fetch teammate checkpoints into the repo from sources listed in manifest.json.

    python -m rlbot.tournament.download              # fetch all configured bots
    python -m rlbot.tournament.download nachi marco  # fetch specific owners
    python -m rlbot.tournament.download --force       # re-fetch even if present
    python -m rlbot.tournament.download --list        # show manifest status, fetch nothing

Each manifest entry has a `type`:
  - url:   http[s] link to a .zip / .tar(.gz) archive of the checkpoint folder
  - git:   git repo URL, optionally '<url>#<subdir>' to copy just a subfolder
  - local: a path on this machine to copy from

Only stdlib is used (urllib, zipfile, tarfile, subprocess). Fetches are idempotent:
an owner that already resolves to a checkpoint is skipped unless --force.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .roster import REPO_ROOT, resolve_checkpoint

MANIFEST = Path(__file__).resolve().parent / "manifest.json"


def _load_manifest() -> dict[str, dict]:
    data = json.loads(MANIFEST.read_text())
    return data["bots"]


def _abs(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def _looks_like_checkpoint(d: Path) -> bool:
    return (d / "PPO_POLICY.pt").is_file() or resolve_checkpoint(str(d)) is not None


def _extract_archive(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip" or zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unrecognised archive format: {archive.name}")


def _fetch_url(source: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "ckpt_archive"
        print(f"    downloading {source} ...")
        urllib.request.urlretrieve(source, archive)
        _extract_archive(archive, dest)


def _fetch_git(source: str, dest: Path) -> None:
    url, _, subdir = source.partition("#")
    with tempfile.TemporaryDirectory() as tmp:
        print(f"    cloning {url} ...")
        subprocess.run(["git", "clone", "--depth", "1", url, tmp], check=True)
        src = Path(tmp) / subdir if subdir else Path(tmp)
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.name == ".git":
                continue
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)


def _fetch_local(source: str, dest: Path) -> None:
    src = _abs(source)
    if not src.exists():
        raise FileNotFoundError(f"local source not found: {src}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


_FETCHERS = {"url": _fetch_url, "git": _fetch_git, "local": _fetch_local}


def fetch_owner(owner: str, entry: dict, *, force: bool) -> str:
    """Fetch one owner's checkpoint. Returns a short status string."""
    dest = _abs(entry["dest"])
    source = entry.get("source", "").strip()

    if not source:
        return "no source in manifest (fill it in)"
    if not force and _looks_like_checkpoint(dest):
        return f"already present at {entry['dest']} (use --force to re-fetch)"

    fetcher = _FETCHERS.get(entry["type"])
    if fetcher is None:
        return f"unknown type {entry['type']!r}"

    fetcher(source, dest)
    if _looks_like_checkpoint(dest):
        return f"OK -> {entry['dest']}"
    return (f"fetched into {entry['dest']} but no PPO_POLICY.pt found — "
            "check the archive layout")


def main() -> None:
    p = argparse.ArgumentParser(description="Download teammate checkpoints from manifest.json.")
    p.add_argument("owners", nargs="*", help="Specific owners to fetch (default: all).")
    p.add_argument("--force", action="store_true", help="Re-fetch even if already present.")
    p.add_argument("--list", action="store_true", help="Show status only; fetch nothing.")
    args = p.parse_args()

    manifest = _load_manifest()
    owners = args.owners or list(manifest)

    print(f"Manifest: {MANIFEST}")
    for owner in owners:
        entry = manifest.get(owner)
        if entry is None:
            print(f"  {owner:<7} not in manifest")
            continue
        dest = _abs(entry["dest"])
        present = _looks_like_checkpoint(dest)
        if args.list:
            src = entry.get("source", "").strip() or "(no source)"
            print(f"  {owner:<7} present={present!s:<5} type={entry['type']:<5} {src}")
            continue
        print(f"  {owner:<7} ...")
        try:
            status = fetch_owner(owner, entry, force=args.force)
        except Exception as exc:
            status = f"FAILED: {exc}"
        print(f"  {owner:<7} {status}")


if __name__ == "__main__":
    main()
