"""Fetch the pinned public checkpoint, sensor archive, or command files."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
TALK2CAR_RELEASE = "3fde52aee3341312196ab9b2f999337eff0693a0"
COMMAND_FILES = {
    "train": (4_891_514, "5a90c471284685bd8dd8bfa6e992cfefce69e58a392c934d2f9d54ea119a42d2"),
    "val": (679_386, "56e2dc27ec1fd83e5e6fb910c6582a7331c591babd194257c8d9680ad22a3f9d"),
}


def asset(kind):
    cfg = json.loads((HERE / "PROTOCOL.json").read_text())
    if kind == "wilddet3d":
        p = cfg[kind]
        filename = p["checkpoint"]
        address = f"https://huggingface.co/{p['checkpoint_repo']}/resolve/{p['checkpoint_revision']}/{filename}"
        return filename, address, p["checkpoint_bytes"], p["checkpoint_sha256"]
    p = cfg["external_data"]
    return "v1.0-mini.tgz", p["archive"], p["expected_bytes"], None


def range_tail(url, partial, expected_size, streams, kind):
    # Each worker owns a disjoint range and resumes only its own bytes.
    start = partial.stat().st_size if partial.exists() else 0
    plan_path = partial.with_suffix(".ranges.json")
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan["prefix_bytes"] == start and plan["asset_bytes"] == expected_size
        spans = plan["spans"]
    else:
        width = (expected_size - start + streams - 1) // streams
        spans = [(lo, min(expected_size, lo + width)-1) for lo in range(start, expected_size, width)]
        plan_path.write_text(json.dumps(dict(prefix_bytes=start, asset_bytes=expected_size, spans=spans)))

    def block(span):
        lo, hi = span
        file = partial.with_suffix(f".range_{lo}")
        done = file.stat().st_size if file.exists() else 0
        assert done <= hi - lo + 1
        if done == hi - lo + 1:
            return file
        headers = {"Range": f"bytes={lo + done}-{hi}"}
        try:
            response = urlopen(Request(url, headers=headers), timeout=90)
        except HTTPError as e:
            raise RuntimeError(f"Public range returned HTTP {e.code}") from None
        with response, file.open("ab") as stream:
            assert response.status == 206
            assert response.headers["Content-Range"] == f"bytes {lo + done}-{hi}/{expected_size}"
            shutil.copyfileobj(response, stream, length=4*2**20)
        assert file.stat().st_size == hi - lo + 1
        return file

    with ThreadPoolExecutor(max_workers=streams) as pool:
        running = [pool.submit(block, span) for span in spans]
        for job in as_completed(running):
            print(kind, "range complete", job.result().stat().st_size // 2**20, "MiB", flush=True)
    join = partial.with_suffix(".assembled")
    with join.open("wb") as fp:
        sources = ([partial] if start else []) + [job.result() for job in running]
        for source in sources:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, fp, length=8*2**20)
    assert join.stat().st_size == expected_size
    join.replace(partial)
    for job in running:
        job.result().unlink()
    plan_path.unlink()


def rebalance(kind, root, tile_mib):
    """Keep every completed byte and split unfinished spans before resuming."""
    filename, _, expected, _ = asset(kind)
    partial = root / (filename + ".part")
    plan_path = partial.with_suffix(".ranges.json")
    plan = json.loads(plan_path.read_text())
    assert plan["asset_bytes"] == expected and tile_mib > 0
    step = tile_mib * 2**20
    spans = []
    begin = plan["prefix_bytes"]
    for start, end in plan["spans"]:
        assert start == begin <= end
        out = partial.with_suffix(f".range_{start}")
        used = out.stat().st_size if out.exists() else 0
        first_missing = start + used
        assert first_missing <= end + 1
        if used:
            spans.append((start, first_missing - 1))
        spans.extend((offset, min(end, offset + step - 1)) for offset in range(first_missing, end + 1, step))
        begin = end + 1
    assert begin == expected and sum(b-a+1 for a, b in spans) + plan["prefix_bytes"] == expected
    plan["spans"] = spans
    swap = plan_path.with_suffix(".tmp")
    swap.write_text(json.dumps(plan))
    swap.replace(plan_path)


def run(kind, root, streams):
    filename, url, expected_size, expected_sha = asset(kind)
    root.mkdir(parents=True, exist_ok=True)
    ready = root / filename
    partial = root / (filename + ".part")
    start = perf_counter()
    if streams > 1 and not ready.exists() and (not partial.exists() or partial.stat().st_size < expected_size):
        range_tail(url, partial, expected_size, streams, kind)
    if not ready.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        assert offset <= expected_size
        if offset != expected_size:
            request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            try:
                response = urlopen(request, timeout=90)
            except HTTPError as e:
                raise RuntimeError(f"Public asset returned HTTP {e.code}") from None
            with response:
                if offset and response.status != 206:
                    raise RuntimeError("Resume requires a partial response")
                if offset:
                    expected_range = f"bytes {offset}-{expected_size - 1}/{expected_size}"
                    assert response.headers.get("Content-Range") == expected_range
                assert int(response.headers.get("Content-Length", expected_size - offset)) == expected_size - offset
                progress = offset + 128 * 2**20
                with partial.open("ab") as stream:
                    while True:
                        block = response.read(4 * 2**20)
                        if not block:
                            break
                        offset += len(block)
                        if offset > expected_size:
                            raise RuntimeError("Transport exceeded the fixed asset length")
                        stream.write(block)
                        if offset >= progress:
                            print(kind, offset // 2**20, "/", expected_size // 2**20, "MiB", flush=True)
                            progress += 128 * 2**20
        assert partial.stat().st_size == expected_size
    path = ready if ready.exists() else partial
    assert path.stat().st_size == expected_size
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_sha is not None:
        assert sha == expected_sha
    else:
        with path.open("rb") as stream:
            assert stream.read(2) == b"\x1f\x8b"
    if path != ready:
        partial.replace(ready)
    record = dict(asset=filename, source=url, bytes=expected_size, sha256=sha, seconds=perf_counter() - start)
    (root / (filename + ".receipt.json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def commands(root):
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for split, (size, digest) in COMMAND_FILES.items():
        filename = f"{split}_commands.json"
        source = f"https://raw.githubusercontent.com/talk2car/Talk2Car/{TALK2CAR_RELEASE}/data/commands/{filename}"
        path = root / filename
        if path.exists():
            payload = path.read_bytes()
        else:
            with urlopen(source, timeout=45) as response:
                payload = response.read(size + 1)
        assert len(payload) == size and hashlib.sha256(payload).hexdigest() == digest
        if not path.exists():
            partial = path.with_suffix(".part")
            partial.write_bytes(payload)
            partial.replace(path)
        sources[split] = dict(filename=filename, source=source, bytes=size, sha256=digest)
    (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("wilddet3d", "nuscenes", "talk2car"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--rebalance-mib", type=int, default=0, help="retile stopped ranges before a multi-stream resume")
    options = parser.parse_args()
    if options.kind == "talk2car":
        commands(options.root)
    else:
        if options.rebalance_mib:
            assert options.streams > 1
            rebalance(options.kind, options.root, options.rebalance_mib)
        run(options.kind, options.root, options.streams)
