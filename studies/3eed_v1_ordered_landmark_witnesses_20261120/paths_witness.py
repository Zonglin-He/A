"""Paths, serialization and stable public identifiers for the experiment."""
from pathlib import Path
from hashlib import sha256
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAST = HERE.parent / '3eed_v1_reference_transposition_20261120'
sys.path.insert(0, str(LAST))
from registry_swap import SOURCES, read, save, digest

WORK = HERE / 'private'
OUT = HERE / 'results'
OBS = WORK / 'observed'
PRED = WORK / 'predictions'
SPEC = HERE / 'PROTOCOL.json'

def tag(s):
    return sha256(('ordered-witness-1|' + str(s)).encode()).hexdigest()[:20]

def name(row):
    return row['uid'] + '_' + row['source'] + '.npz'
