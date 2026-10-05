#!/usr/bin/env python3
"""Download a reproducible HC-STVG2 validation subset from the official release.

The official OneDrive release stores videos in ten ~11 GB ZIP archives. This
script uses HTTP Range requests to read each ZIP central directory and extract
only the selected members; it does not download all 115 GB.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, urljoin

import numpy as np
import requests
from bs4 import BeautifulSoup
from remotezip import RemoteZip


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARE_URL = (
    "https://intxyz-my.sharepoint.com/:f:/g/personal/"
    "zongheng_picdataset_com/ErqA01jikPZKnudZe6-Za9MBe17XXAxJr9ODn65Z2qGKkw"
    "?e=7vKw1U"
)
DEFAULT_PASSWORD = "tzhhhh123"  # Publicly documented in the HC-STVG repository.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", default=str(PROJECT_ROOT / "data" / "hcstvg2_subset")
    )
    parser.add_argument(
        "--metadata-cache",
        default=str(PROJECT_ROOT / "data" / "hcstvg2_official_metadata"),
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="subset manifest whose files (and optionally source clusters) are excluded",
    )
    parser.add_argument(
        "--exclude-filename",
        action="append",
        default=[],
        help="specific annotation/video filename to exclude after a documented data audit",
    )
    parser.add_argument(
        "--exclusion-note",
        default="",
        help="human-readable reason for explicit filename exclusions",
    )
    parser.add_argument(
        "--exclude-source-clusters",
        action="store_true",
        help="exclude every clip from source-video clusters present in excluded manifests",
    )
    parser.add_argument(
        "--max-per-source-cluster",
        type=int,
        default=0,
        help="optional cap on selected clips sharing the same underlying source video",
    )
    parser.add_argument(
        "--repair-existing",
        action="store_true",
        help="retain valid members of the existing subset and replace explicit exclusions",
    )
    return parser.parse_args()


def source_cluster(filename: str) -> str:
    """Return the underlying source-video id from HC-STVG2 clip filenames."""

    stem = Path(filename).stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def authenticate(password: str) -> tuple[requests.Session, str, str]:
    session = requests.Session()
    response = session.get(SHARE_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    form = soup.find("form", id="inputForm")
    if form is None:
        raise RuntimeError("official OneDrive password form was not found")
    payload = {
        item.get("name"): item.get("value", "")
        for item in form.find_all("input")
        if item.get("name")
    }
    payload.update(txtPassword=password, btnSubmitPassword="Verify")
    response = session.post(
        urljoin(response.url, form["action"]),
        data=payload,
        timeout=30,
    )
    response.raise_for_status()
    if 'id="txtPassword"' in response.text:
        raise PermissionError("the official HC-STVG OneDrive password was rejected")

    import re

    drive_match = re.search(r'"\.driveUrl":"([^"]+)"', response.text)
    token_match = re.search(r'"\.driveAccessToken":"([^"]+)"', response.text)
    if drive_match is None or token_match is None:
        raise RuntimeError("could not resolve the official OneDrive API metadata")
    return session, drive_match.group(1), token_match.group(1)


def list_children(
    session: requests.Session, drive_url: str, token: str, folder: str
) -> list[dict]:
    url = f"{drive_url}/root:/{quote(folder, safe='/')}:/children?{token}"
    response = session.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "@odata.nextLink" in payload:
        raise RuntimeError("unexpected paginated OneDrive folder listing")
    return payload["value"]


def download_file(
    session: requests.Session, item: dict, destination: Path
) -> None:
    expected_size = int(item["size"])
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    url = item.get("@content.downloadUrlNoAuth") or item.get("@content.downloadUrl")
    if not url:
        raise RuntimeError(f"no content URL for official file {item['name']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    if temporary.stat().st_size != expected_size:
        raise IOError(f"incomplete download for {item['name']}")
    temporary.replace(destination)


def crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            value = zlib.crc32(chunk, value)
    return value & 0xFFFFFFFF


def extract_selected_members(
    session: requests.Session,
    archive_item: dict,
    selected_names: list[str],
    video_directory: Path,
) -> list[dict]:
    url = archive_item.get("@content.downloadUrl")
    if not url:
        raise RuntimeError(f"no authenticated URL for {archive_item['name']}")
    extracted: list[dict] = []
    with RemoteZip(url, session=session) as archive:
        members = {
            Path(info.filename).name: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        missing = sorted(set(selected_names) - set(members))
        if missing:
            raise FileNotFoundError(
                f"{archive_item['name']} is missing selected members: {missing}"
            )
        for name in selected_names:
            info = members[name]
            destination = video_directory / name
            valid_existing = (
                destination.is_file()
                and destination.stat().st_size == info.file_size
                and crc32_file(destination) == info.CRC
            )
            if not valid_existing:
                video_directory.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".part")
                with archive.open(info) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
                if temporary.stat().st_size != info.file_size:
                    raise IOError(f"incomplete ZIP member: {name}")
                if crc32_file(temporary) != info.CRC:
                    raise IOError(f"CRC mismatch for ZIP member: {name}")
                temporary.replace(destination)
            extracted.append(
                {
                    "filename": name,
                    "archive": archive_item["name"],
                    "size_bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                }
            )
            print(
                f"[{len(extracted)}/{len(selected_names)}] "
                f"{archive_item['name']} -> {name} ({info.file_size / 1024**2:.1f} MiB)",
                flush=True,
            )
    return extracted


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive")
    output_root = Path(args.output_root).expanduser().resolve()
    metadata_cache = Path(args.metadata_cache).expanduser().resolve()
    annotation_directory = output_root / "annotations"
    video_directory = output_root / "video"

    session, drive_url, token = authenticate(args.password)
    annotation_items = list_children(
        session, drive_url, token, "HC-STVG/anno_v2"
    )
    video_items = list_children(session, drive_url, token, "HC-STVG/VIdeo")
    item_by_name = {item["name"]: item for item in annotation_items + video_items}
    for name in ("val_v2.json", "video_parts.json"):
        if name not in item_by_name:
            raise FileNotFoundError(f"official OneDrive release has no {name}")
        download_file(session, item_by_name[name], metadata_cache / name)

    with (metadata_cache / "val_v2.json").open("r") as handle:
        annotations: dict[str, dict] = json.load(handle)
    with (metadata_cache / "video_parts.json").open("r") as handle:
        parts: dict[str, list[str]] = json.load(handle)
    excluded_filenames: set[str] = set()
    excluded_clusters: set[str] = set()
    for manifest_path in args.exclude_manifest:
        with Path(manifest_path).expanduser().resolve().open("r") as handle:
            excluded_manifest = json.load(handle)
        manifest_filenames = {
            sample["filename"] for sample in excluded_manifest.get("samples", [])
        }
        excluded_filenames.update(manifest_filenames)
        excluded_clusters.update(source_cluster(name) for name in manifest_filenames)
    excluded_filenames.update(args.exclude_filename)
    file_to_part = {
        filename: part for part, filenames in parts.items() for filename in filenames
    }
    missing_part = sorted(set(annotations) - set(file_to_part))
    if missing_part:
        raise RuntimeError(
            f"{len(missing_part)} validation videos are absent from video_parts.json"
        )

    filenames = list(annotations)
    rng = np.random.default_rng(args.seed)
    eligible_indices = [
        index
        for index, filename in enumerate(filenames)
        if filename not in excluded_filenames
        and (
            not args.exclude_source_clusters
            or source_cluster(filename) not in excluded_clusters
        )
    ]
    existing_manifest_path = output_root / "subset_manifest.json"
    if args.repair_existing and existing_manifest_path.is_file():
        with existing_manifest_path.open("r") as handle:
            existing_manifest = json.load(handle)
        existing_selected = [sample["filename"] for sample in existing_manifest["samples"]]
        retained = [
            name
            for name in existing_selected
            if name in annotations
            and name not in excluded_filenames
            and (
                not args.exclude_source_clusters
                or source_cluster(name) not in excluded_clusters
            )
        ]
        retained_set = set(retained)
        cluster_counts: dict[str, int] = defaultdict(int)
        for name in retained:
            cluster_counts[source_cluster(name)] += 1
        chosen = [filenames.index(name) for name in retained]
        for index in rng.permutation(eligible_indices).tolist():
            name = filenames[index]
            cluster = source_cluster(name)
            if name in retained_set:
                continue
            if (
                args.max_per_source_cluster > 0
                and cluster_counts[cluster] >= args.max_per_source_cluster
            ):
                continue
            cluster_counts[cluster] += 1
            chosen.append(index)
            if len(chosen) == args.num_samples:
                break
        source_indices = sorted(chosen)
    elif args.max_per_source_cluster > 0:
        cluster_counts: dict[str, int] = defaultdict(int)
        chosen: list[int] = []
        for index in rng.permutation(eligible_indices).tolist():
            cluster = source_cluster(filenames[index])
            if cluster_counts[cluster] >= args.max_per_source_cluster:
                continue
            cluster_counts[cluster] += 1
            chosen.append(index)
            if len(chosen) == args.num_samples:
                break
        source_indices = sorted(chosen)
    else:
        if args.num_samples > len(eligible_indices):
            raise ValueError(
                f"requested {args.num_samples} samples from {len(eligible_indices)} eligible annotations"
            )
        source_indices = sorted(
            rng.choice(eligible_indices, size=args.num_samples, replace=False).tolist()
        )
    if len(source_indices) != args.num_samples:
        raise ValueError(
            f"cluster cap permits only {len(source_indices)} of {args.num_samples} requested samples"
        )
    selected = [filenames[index] for index in source_indices]
    selected_by_part: dict[str, list[str]] = defaultdict(list)
    for name in selected:
        selected_by_part[file_to_part[name]].append(name)

    extracted: list[dict] = []
    for part in sorted(selected_by_part, key=int):
        archive_name = f"{part}.zip"
        if archive_name not in item_by_name:
            raise FileNotFoundError(f"official OneDrive release has no {archive_name}")
        records = extract_selected_members(
            session,
            item_by_name[archive_name],
            selected_by_part[part],
            video_directory,
        )
        extracted.extend(records)

    annotation_directory.mkdir(parents=True, exist_ok=True)
    selected_annotations = {name: annotations[name] for name in selected}
    with (annotation_directory / "valv2.json").open("w") as handle:
        json.dump(selected_annotations, handle)
    records_by_name = {record["filename"]: record for record in extracted}
    manifest = {
        "source": "official HC-STVG2 password-protected OneDrive release",
        "share_url": SHARE_URL,
        "official_validation_size": len(annotations),
        "selection_seed": args.seed,
        "num_samples": len(selected),
        "source_indices": source_indices,
        "excluded_manifests": [
            str(Path(path).expanduser().resolve()) for path in args.exclude_manifest
        ],
        "explicitly_excluded_filenames": sorted(args.exclude_filename),
        "exclusion_note": args.exclusion_note,
        "exclude_source_clusters": args.exclude_source_clusters,
        "max_per_source_cluster": args.max_per_source_cluster,
        "repaired_existing_subset": args.repair_existing,
        "num_source_clusters": len({source_cluster(name) for name in selected}),
        "source_clusters": sorted({source_cluster(name) for name in selected}),
        "total_downloaded_video_bytes": sum(
            records_by_name[name]["size_bytes"] for name in selected
        ),
        "samples": [
            {
                "source_index": source_index,
                **records_by_name[name],
            }
            for source_index, name in zip(source_indices, selected)
        ],
    }
    with (output_root / "subset_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        f"downloaded and CRC-verified {len(selected)} official validation videos "
        f"({manifest['total_downloaded_video_bytes'] / 1024**3:.2f} GiB) to {output_root}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
