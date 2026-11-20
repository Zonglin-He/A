from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / '3eed_v1_paired_description_subject_readout_20261120'
sys.path.insert(0, str(PREV))
import paired as legacy
from paired import affine, anon, archive_npz, atomic, capture, digest, engine, fixed_prior, np, read, torch

WORK = HERE / 'run_PRIVATE'
RESULTS = HERE / 'results'
ARMS = tuple(read(HERE / 'PROTOCOL.json')['arms'])


def setup():
    legacy.initialize()
    WORK.mkdir(mode=0o700, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)


def pin(path, content):
    if path.exists():
        if read(path) != content:
            raise ValueError(f'Dependency changed after registration: {path}')
    else:
        atomic(path, content)
