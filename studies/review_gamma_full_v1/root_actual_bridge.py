"""Root explicitly checks retained first paid raw grids before allowing disposal."""
from gamma_full_common import *
import argparse
def run(ds):
    import torch,numpy as np
    from gamma_cached_core_v1 import build
    from stvg.representation import extract
    from stvg.spatial import SpatialGeometry,evidence_dp
    review.gt_guard();torch.set_num_threads(4);validate();wait=read(D/f'{ds}_bridge_waiting.json')
    assert wait['status']=='two_actual_paid_raw_commits_waiting_root_repool_check'
    assert not(D/f'{ds}_root_actual_bridge.json').exists()
    rows={r['id']:r for r in records(ds)};proof=[]
    for b in wait['files']:
        key=b['query'];assert sha(b['raw_path'])==b['raw_sha256']and sha(b['receipt_path'])==b['receipt_sha256']
        rec=read(b['receipt_path']);assert rec['raw_origin']['before_teacher_or_ROI_pool']and rec['teacher']['actual_eight_RGB_context_exact']
        c,parent=selected.suite().load_parent(ds,rows[key]);assert parent==rec['parent'];current,ce=checked_current(ds,key)
        raw=torch.load(b['raw_path'],map_location='cpu',weights_only=False,mmap=True)
        geo=SpatialGeometry.from_cached_geometry(c['geometry']);aux=evidence_dp(geo.for_interval([c['frame_ids'][0],c['frame_ids'][-1]+1]))['final'].numpy()
        f,_=extract(raw,aux);assert np.array_equal(f,c['content_features'])
        a,compact,default=build(c,raw,ds,current);again,_,_=build(c,raw,ds,current)
        assert all(np.array_equal(a[k],again[k])for k in a)
        with np.load(D/rec['path'])as z:assert all(np.array_equal(a[k],z[k])for k in a)
        old=torch.load(D/rec['compact_path'],map_location='cpu',weights_only=False,mmap=True);cold=complete_arrays(old,current)
        assert all(np.array_equal(a[k],cold[k])for k in a)
        proof.append(dict(raw_SHA256=b['raw_sha256'],teacher_SHA256=rec['teacher']['payload_sha256'],actual_old_aux_ROI_pool_exact=True,actual_five_gamma_E_repool_and_all_arrays_repeat_exact=True,committed_compact_cold_arrays_exact=True,default_arm=default,native_proof=rec['native_proof']))
    assert len(proof)==2 and not torch.cuda.is_initialized()
    ack=dict(status='passed_actual_new_raw_bridge',files=wait['files'],proof=proof,worker=identity(),new_Source=0,new_expert=0,new_GT_scoring=0,GT_read=False)
    atomic(D/f'{ds}_root_actual_bridge.json',ack);atomic(P/f'{ds}_actual_new_raw_bridge.json',{k:v for k,v in ack.items()if k!='files'})
    print(ack['status'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dataset',choices=COUNTS,required=True);run(a.parse_args().dataset)
