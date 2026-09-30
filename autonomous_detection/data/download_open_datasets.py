"""Download the open Indian / South Asian traffic datasets that replace IDD.

WHY THIS EXISTS
---------------
The IDD portal became unreachable, and two of the four Mendeley datasets the
problem statement's search turns up -- Indistreet2K25 and IndiaScene365 -- are
under embargo until 2030. What remains downloadable, without registration and
under CC BY 4.0, is enough to cover all six decision groups:

    UVH-26          26,646 images   Indian vehicles, 14 fine-grained types
                                    (Hugging Face, COCO JSON)
    HeteroTraffic   16,289 images   17 classes incl. PEDESTRIAN and a wide
                                    spread of South Asian vehicles (YOLO txt)
    DATS_2022        1,715 annots   Indian roads, incl. CATTLE, goats, dogs,
                                    camels, horses, bullock carts (VOC XML)

UVH-26 alone has no pedestrians or animals, so the VULNERABLE group -- the one
the safety argument rests on -- would never be exercised. The other two supply
it.

RUN IT ON THE LOGIN NODE. Compute nodes usually have no outbound internet.

    python data/download_open_datasets.py            # all three
    python data/download_open_datasets.py --only dats hetero
    python data/download_open_datasets.py --list     # sizes, download nothing

Resumable: a file already on disk with the expected size is skipped, so an
interrupted run can simply be started again. HeteroTraffic's archive is checked
against the SHA-256 Mendeley publishes before it is unpacked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA = PROJECT_ROOT / "data"

MENDELEY = "https://data.mendeley.com/public-api/datasets"
# Mendeley's file host answers 403 to urllib's default user agent.
HEADERS = {"User-Agent": "curl/8.0"}

DATASETS = {
    "hetero": dict(name="HeteroTraffic", mendeley_id="tvkfk56s2k", version=2,
                   out="HeteroTraffic"),
    "dats": dict(name="DATS_2022", mendeley_id="nfc34n8svj", version=2,
                 out="DATS_2022"),
    "uvh26": dict(name="UVH-26", hf_repo="iisc-aim/UVH-26", out="UVH26"),
}


def _open(url: str, timeout: int = 60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS),
                                  timeout=timeout)


def _json(url: str):
    with _open(url) as r:
        return json.loads(r.read())


def mendeley_files(ds_id: str, version: int) -> list:
    """Every file in a Mendeley dataset, with its folder path.

    The public API lists root files and folders separately, and a folder's
    files only by its id -- so the folder tree is rebuilt from parent ids and
    walked. Files at the root carry an empty path.
    """
    out = []
    for f in _json(f"{MENDELEY}/{ds_id}/files?folder_id=root&version={version}"):
        out.append(("", f))

    folders = _json(f"{MENDELEY}/{ds_id}/folders/{version}")
    by_id = {f["id"]: f for f in folders}

    def path_of(fid: str) -> str:
        parts, seen = [], set()
        while fid in by_id and fid not in seen:
            seen.add(fid)
            parts.append(by_id[fid]["name"])
            fid = by_id[fid].get("parent_id", "")
        return "/".join(reversed(parts))

    for fid in by_id:
        for f in _json(f"{MENDELEY}/{ds_id}/files?folder_id={fid}&version={version}"):
            out.append((path_of(fid), f))
    return out


def _fetch(url: str, dest: Path, size: int, retries: int = 4) -> str:
    """Download one file, skipping it if already complete."""
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with _open(url, timeout=120) as r, open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            if size and tmp.stat().st_size != size:
                raise IOError(f"size {tmp.stat().st_size} != expected {size}")
            tmp.replace(dest)
            return "ok"
        except Exception as exc:                            # noqa: BLE001
            if attempt == retries:
                return f"FAILED: {exc}"
            time.sleep(2 * attempt)
    return "FAILED"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download_mendeley(key: str, list_only: bool) -> bool:
    spec = DATASETS[key]
    root = DATA / spec["out"]
    files = mendeley_files(spec["mendeley_id"], spec["version"])
    total = sum(f["size"] for _, f in files)
    print(f"\n{spec['name']}: {len(files)} files, {total / 1e9:.2f} GB -> {root}")
    if list_only:
        return True

    jobs = []
    for folder, f in files:
        dest = root / folder / f["filename"] if folder else root / f["filename"]
        jobs.append((f["content_details"]["download_url"], dest, f["size"],
                     f["content_details"].get("sha256_hash")))

    results = {"ok": 0, "skip": 0, "failed": []}
    # Many small XML/image files: fetch in parallel. One large archive: the
    # pool simply runs it on a single worker.
    with ThreadPoolExecutor(max_workers=8) as ex:
        for (url, dest, size, _), status in zip(
                jobs, ex.map(lambda j: _fetch(j[0], j[1], j[2]), jobs)):
            if status == "ok":
                results["ok"] += 1
            elif status == "skip":
                results["skip"] += 1
            else:
                results["failed"].append((dest.name, status))
    print(f"  downloaded {results['ok']}, already present {results['skip']}, "
          f"failed {len(results['failed'])}")
    for name, why in results["failed"][:10]:
        print(f"    {name}: {why}")

    # Verify and unpack archives. A truncated 4.9 GB zip that unpacks halfway
    # produces a dataset missing a random slice of images -- silently.
    for url, dest, size, sha in jobs:
        if dest.suffix.lower() != ".zip" or not dest.exists():
            continue
        if sha:
            print(f"  verifying {dest.name} (sha256) ...")
            if _sha256(dest) != sha:
                print(f"  FAILED: {dest.name} does not match the published "
                      f"checksum. Delete it and re-run.")
                return False
        marker = dest.with_suffix(".unpacked")
        if marker.exists():
            print(f"  {dest.name} already unpacked")
            continue
        print(f"  unpacking {dest.name} ...")
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.parent)
        marker.write_text("ok")
    return not results["failed"]


def download_uvh26(list_only: bool) -> bool:
    spec = DATASETS["uvh26"]
    root = DATA / spec["out"]
    print(f"\nUVH-26: Hugging Face {spec['hf_repo']} -> {root}")
    if list_only:
        return True
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub missing: python -m pip install huggingface_hub")
        return False
    snapshot_download(repo_id=spec["hf_repo"], repo_type="dataset",
                      local_dir=str(root), max_workers=8)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(DATASETS),
                    help="download only these (default: all three)")
    ap.add_argument("--list", action="store_true",
                    help="print file counts and sizes; download nothing")
    args = ap.parse_args()

    keys = args.only or ["dats", "hetero", "uvh26"]
    ok = True
    for key in keys:
        try:
            ok &= (download_uvh26(args.list) if key == "uvh26"
                   else download_mendeley(key, args.list))
        except Exception as exc:                           # noqa: BLE001
            print(f"\n{DATASETS[key]['name']}: FAILED -- {exc}")
            print("  If this is a connection error you are probably on a "
                  "compute node. Run on the login node.")
            ok = False

    if not args.list:
        print("\nNext:")
        print("  python data/prepare_indian.py --src data/DATS_2022     --report-only")
        print("  python data/prepare_indian.py --src data/HeteroTraffic --report-only")
        print("  qsub training/pbs/granularity_step1_prepare.pbs")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
