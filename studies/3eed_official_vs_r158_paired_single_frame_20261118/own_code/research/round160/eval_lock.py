"""Check the task evaluation and benchmark sidecar code pinned before scoring."""
from common160 import HERE, ROOT, SPEC, read, sha


def verify():
    spec=read(HERE/'experiment/EVALUATOR_LOCK.json')
    assert spec['protocol_sha256']==sha(SPEC)
    for name,value in spec['sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('Locked evaluator input changed: '+name)
