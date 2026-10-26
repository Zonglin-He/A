"""Root validates the full paid Source checkpoint as well as actual raw repooling."""
from gamma_full_common import *
from runtime_common_v2 import validate_runtime
import argparse
def run(ds):
    import torch,numpy as np
    import root_actual_bridge as frozen
    validate_runtime();review.gt_guard();torch.set_num_threads(1)
    rows={r['id']:r for r in records(ds)}
    for b in read(D/f'{ds}_bridge_waiting.json')['files']:
        source=b['Source_checkpoint']
        if source is None:continue  # Native prefix inherits the separately SHA-bound native heads.
        assert source['before_native_assertions_and_spatial_extraction']and source['before_teacher_or_ROI_pool']
        assert sha(source['path'])==source['sha256']
        raw=torch.load(source['path'],map_location='cpu',weights_only=False,mmap=True)
        parent,_=selected.suite().load_parent(ds,rows[b['query']])
        assert not raw['GT_read']and raw['frame_ids']==parent['frame_ids']
        assert np.array_equal(raw['Source'],parent['Source'])
        assert len(raw['native_logits'])==2 and all(torch.equal(a,z)for a,z in zip(raw['native_logits'],parent['native_logits']))
        assert [x['frame_ids']for x in raw['records']]==[x['frame_ids']for x in parent['records']]
    frozen.run(ds)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dataset',choices=COUNTS,required=True);run(a.parse_args().dataset)
