"""Locked inputs and arm names for the ordinary, noun and historical panels."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '3eed_v1_noun_vote_transport_20261120'))
import vote_paths as prior
from vote_paths import ROOT, SOURCES, digest, read, save, archive_npz, row_id

SPEC = HERE / 'PROTOCOL.json'
CFG = read(SPEC)
ARMS = tuple(CFG['arms'])
WORK = HERE / 'private'
OUT = HERE / 'results'


def catalog():
    receipt = prior.WORK / 'CATALOG.json'
    assert digest(receipt) == CFG['input_catalog_receipt_sha256']
    seal = read(receipt)
    assert not seal['labels_used'] and digest(prior.WORK/'catalog.json') == seal['catalog']
    assert digest(ROOT/'crossplatform/single_frame_sources.json') == CFG['source_local_weights_sha256']
    assert digest(prior.FULL/'predictions/SEALED.json') == CFG['ordinary_seal']
    return read(prior.WORK/'catalog.json')


def pins():
    return {n:digest(HERE/n) for n in ('common.py','PROTOCOL.json','predict.py')}
