"""Check persistent one-expert repeatability; no student or task scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--output', default='results/tubedetr/teacher_service_check.json')
    args = p.parse_args()
    from stvg.runtime import setup, lease, sha
    from stvg.inputs import validate_metadata
    from stvg.tube_decode import decode
    from stvg.teacher_service import PersistentRefer
    from stvg.tube_runner import write
    request = json.loads(Path(args.request).read_text())
    validate_metadata(request['metadata'])
    setup()
    lock = lease()
    teacher = None
    try:
        frames, ids = decode(request['metadata'], request['dataset'])
        # Fixed uniform positions check the service only; these are not a claim
        # of a native-Source-selected method run or a formal query prediction.
        positions = np.linspace(0, len(ids)-1, 8).astype(int).tolist()
        observed = frames[positions].copy()
        del frames
        teacher = PersistentRefer(ROOT/'logs/teacher_service_check.log')
        first = teacher(observed, request['metadata']['caption'])
        second = teacher(observed, request['metadata']['caption'])
        if not np.array_equal(first['boxes'], second['boxes']) or not np.array_equal(first['native_query_scores'], second['native_query_scores']):
            raise RuntimeError('Persistent expert output changes across identical independent inputs')
        if not first['load_audit']['all_learned_parameters_covered'] or first['load_audit']['independent_detector_instances'] != 0:
            raise RuntimeError('One-expert checkpoint coverage failed')
        write(args.output, dict(status='pass', scope='expert service repeatability only; no student or accuracy evidence',
            checkpoint_sha256=sha(ROOT/'checkpoints/referdino/ryt_mevis_swinb.pth'), request_sha256=sha(args.request),
            input_frames=8, positions_policy='uniform engineering check, not Source observation policy',
            repeated_calls=2, outputs_bitexact=True, learned_parameters_covered=True,
            expert_instances=1, independent_detector_instances=0,
            boxes_sha256=hashlib.sha256(first['boxes'].tobytes()).hexdigest(),
            load_seconds=teacher.load_seconds, first_call_cost=first['cost'],
            second_call_cost=second['cost'], peak_allocated_bytes=max(first['peak_GPU_bytes'], second['peak_GPU_bytes']),
            GT_read=False, development_exposure=True))
        print('One-expert persistent service check passed', flush=True)
    finally:
        if teacher:
            teacher.close()
        lock.close()


if __name__ == '__main__':
    main()
