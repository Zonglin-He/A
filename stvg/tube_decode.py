"""Existing dataset-specific decoder functions under a pinned local tool build.

The historical TA-STVG executable fingerprint remains untouched. This adapter
registers its own FFmpeg build explicitly, and includes that receipt in every
qualification/run identity. Original-build pixel parity is a separate check.
"""
import json
from pathlib import Path
from .runtime import ROOT, sha


def asset_identity():
    registry = ROOT/'runtime/tubedetr/assets.json'
    value = json.loads(registry.read_text())
    if value['ffmpeg_version'] != '8.0.1' or sha(value['ffmpeg_path']) != value['ffmpeg_sha256']:
        raise ValueError('The registered TubeDETR FFmpeg build changed')
    return value


def decode(metadata, dataset):
    from . import decode as legacy
    asset = asset_identity()
    previous = legacy.FFMPEG, legacy.FFMPEG_SHA
    try:
        legacy.FFMPEG = Path(asset['ffmpeg_path'])
        legacy.FFMPEG_SHA = asset['ffmpeg_sha256']
        return legacy.decode(metadata, dataset)
    finally:
        legacy.FFMPEG, legacy.FFMPEG_SHA = previous
