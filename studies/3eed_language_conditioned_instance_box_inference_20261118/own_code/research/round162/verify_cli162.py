"""Exercise the raw research entry point and verify frozen weights on audited inputs."""
import sys
import time

import numpy as np

from manifest162 import DEV, HERE, ORDER, PLATFORMS, ROOT, WORK, atomic_json, target_rows
sys.path.insert(0,str(ROOT))
from crossplatform.ground_lcib import SingleFrameLCIB, WAYMO_VIEWS
from crossplatform.ground_single_frame import sensor_frame_box


def fingerprint(model):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    import torch
    outcomes = []
    rows = target_rows()
    for source in PLATFORMS:
        model=SingleFrameLCIB(source)
        base, matcher=model.model.base,model.model.matcher
        assert not any(p.requires_grad for p in model.model.base.parameters())
        assert not any(p.requires_grad for p in model.model.matcher.parameters())
        before=fingerprint(base),fingerprint(matcher)
        model_latency=[]
        for s,target in ORDER:
            if s!=source:
                continue
            i, data=next((i,e) for i,e in enumerate(rows) if (e['source'],e['target'],e['cohort'])==(s,target,DEV))
            row=data['r']
            view=int(row['suffix'] or 0)
            inputs=dict(target=target,points=row['pointpath'],rgb=row['imagepath'],calibration=row['metapath'],
                        text=row['caption'],seed=row['seed'],waymo_view=WAYMO_VIEWS[view])
            model.predict(**inputs)
            start=time.perf_counter()
            old=model.model.predict(**inputs)
            original=time.perf_counter()-start
            start=time.perf_counter()
            got=model.predict(**inputs)
            timing=time.perf_counter()-start
            with np.load(WORK/'DEV'/f'{i:05d}.npz',allow_pickle=False) as f:
                want=sensor_frame_box(f['boxes'][f['arms'].tolist().index('lcib')],target,view)
                want_old=sensor_frame_box(f['old_pca'],target,view)
            np.testing.assert_allclose(got,want,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(old,want_old)
            model_latency.append(dict(source=s,target=target,original_ms=1000*original,LCIB_ms=1000*timing))
        assert before==(fingerprint(base),fingerprint(matcher))
        outcomes.extend(model_latency)
        del base,matcher,model
        torch.cuda.empty_cache()
    atomic_json(HERE/'RAW_RUNTIME_AUDIT.json',dict(raw_entry_passed=len(outcomes), before_after_weights_identical=True,
                before_after_buffers_identical=True, timing_subset=outcomes, FINAL_access=False))
    print('Raw research entry verified',len(outcomes),flush=True)


if __name__=='__main__':
    main()
