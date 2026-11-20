"""Fetch pinned public weights with restartable ranges and verify the published hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

import requests


STUDY = Path(__file__).resolve().parent
CHUNK = 16 * (1 << 20)


def check_file(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    if "git_blob_sha1" in spec:
        raw = path.read_bytes()
        return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest() == spec["git_blob_sha1"]
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() == spec["sha256"]


def get_chunk(base, hosts, url_path, name, start, stop, size):
    part = base / f"{name}.part-{start}"
    if part.is_file() and part.stat().st_size == stop - start:
        return part
    temp = part.with_name(part.name + ".tmp")
    expected = f"bytes {start}-{stop - 1}/{size}"
    last_error = None
    for attempt in range(8):
        host = hosts[(start // CHUNK + attempt) % len(hosts)]
        request_headers = {"Range": f"bytes={start}-{stop - 1}", "Accept-Encoding": "identity"}
        try:
            url = f"https://{host}/{url_path}?segment={start}-{stop}"
            with requests.get(url, headers=request_headers, stream=True, timeout=(18, 80)) as resp:
                resp.raise_for_status()
                if resp.status_code != 206 or resp.headers.get("Content-Range") != expected:
                    raise ValueError(f"Invalid range response from {host}")
                with temp.open("wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
            if temp.stat().st_size != stop - start:
                raise ValueError(f"Incorrect piece length from {host}")
            temp.replace(part)
            return part
        except (requests.RequestException, ValueError, OSError) as err:
            last_error = type(err).__name__
            time.sleep(min(5, attempt + 1))
    raise RuntimeError(f"Cannot download {name} at {start}: {last_error}")


def fetch(root, name, model, spec, workers, hosts):
    folder = root / model["asset"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if check_file(target, spec):
        print(f"{target.name} already verified", flush=True)
        return
    url_path = f'{model["repository"]}/resolve/{model["revision"]}/{name}'
    size = spec["size"]
    temp = target.with_name(target.name + ".assembling")
    if size < CHUNK:
        response = requests.get(f"https://huggingface.co/{url_path}", timeout=(18, 80))
        response.raise_for_status()
        temp.write_bytes(response.content)
    else:
        pieces = list(range(0, size, CHUNK))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(get_chunk, folder, hosts, url_path, name, start, min(start + CHUNK, size), size)
                    for start in pieces]
            for count, job in enumerate(as_completed(jobs), 1):
                job.result()
                if count % 20 == 0 or count == len(jobs):
                    print(f"{name} {count}/{len(jobs)} chunks", flush=True)
        with temp.open("wb") as assembled:
            for start in pieces:
                with (folder / f"{name}.part-{start}").open("rb") as piece:
                    shutil.copyfileobj(piece, assembled, CHUNK)
    if not check_file(temp, spec):
        raise RuntimeError(f"Checksum mismatch for {name}")
    temp.replace(target)
    for part in folder.glob(f"{name}.part-*"):
        part.unlink()
    print(f"{name} verified ({size} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_root", type=Path)
    parser.add_argument("asset", choices=("hispatial", "moge", "prior_depth"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--also-mirror", action="store_true")
    args = parser.parse_args()
    model = json.loads((STUDY / "MODELS.json").read_bytes())[args.asset]
    model["asset"] = args.asset
    hosts = ("huggingface.co", "hf-mirror.com") if args.also_mirror else ("huggingface.co",)
    for filename, spec in model["files"].items():
        fetch(args.weights_root, filename, model, spec, args.workers, hosts)
