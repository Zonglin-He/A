"""Separate source-origin binding for the already sealed actual O root audit."""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, sha
import scripts.review_stvg_opd_common_state_postseal_v1 as original
import scripts.review_stvg_opd_common_state_objective002 as matched

REV = original.REC / 'source_origin_stream003'


def run():
    for path, digest in read(REV / 'REVISION_RUNTIME.json')['pins'].items():
        assert sha(ROOT / path) == digest, path
    original.require_seal()
    source = inspect.getsource(original.run)
    key = "(r['dataset'], r['source'], inp['source_model_state_sha256'])"
    assert source.count("if r['dataset'] in origin:") == 1
    assert source.count("origin[r['dataset']]") == 2
    source = source.replace("if r['dataset'] in origin:", 'if ' + key + ' in origin:')
    source = source.replace("origin[r['dataset']]", 'origin[' + key + ']')
    namespace = dict(original.__dict__)
    exec(compile(source, __file__, 'exec'), namespace)
    original.run = namespace['run']
    # Keep the already-qualified Direct-versus-Gaussian objective dispatch.
    matched.run()


if __name__ == '__main__':
    run()
