"""Analytical spatial/temporal gains verify all five cached readouts."""
import hashlib
import json
import numpy as np
import pytest
from stvg.evaluation_v2 import score


def test_five_components_from_same_complete_prediction(tmp_path):
    fids=list(range(10))
    record=dict(id='synthetic',dataset='hc2',key='synthetic',metadata=dict(
        caption='Synthetic.',index=0,width=4,height=4,fps=2.,frame_ids=fids))
    manifest,labels=tmp_path/'input.jsonl',tmp_path/'labels.jsonl'
    manifest.write_text(json.dumps(record)+'\n')
    labels.write_text(json.dumps(dict(id='synthetic',dataset='hc2',parent='parent',
        interval=[0,8],boxes={str(i):[0,0,4,4]for i in range(8)}))+'\n')
    root=tmp_path/'pred';root.mkdir()
    np.savez(root/'one.npz',Source=np.tile([.25,.5,.5,1.],(10,1)),
        E_Ref=np.tile([.375,.5,.75,1.],(10,1)),Live=np.tile([.5,.5,1.,1.],(10,1)),
        frame_ids=fids,positions8=[0,1,2,3,4,5,8,9],Native_interval=[0,4],Strong_interval=[0,8])
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'sealed.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest),queries=1,
        files={'synthetic':dict(path='one.npz',sha256=sha(root/'one.npz'))})))
    arms=['Source','Spatial','Temporal','WithoutOPD','Full']
    result=score(manifest,labels,root,tmp_path/'metrics.json',arms=arms)
    values=result['datasets']['hc2']['query_means']
    for arm,expected in zip(arms,[.25,.5,.5,.75,1.]):
        assert values[arm]['m_vIoU']==pytest.approx(expected)
    assert values['Spatial']['m_tIoU']==.5 and values['Temporal']['m_tIoU']==1.
    assert values['WithoutOPD']['m_sIoU']==.75 and values['Full']['m_sIoU']==1.
    assert values['Source']['boundary_MAE_seconds']==1.
    assert values['Full']['boundary_MAE_seconds']==0.
    assert values['Full']['observed_GT_nodes']==6
    assert values['Full']['unobserved_GT_nodes']==2
    assert values['Full']['unobserved_actual_sIoU']==1.
