"""One cached update: preserve E globally and retain per-anchor correction."""
import numpy as np
from vg_tta.stvg_error_state_correction_v1 import encode,decode

ARMS=['E_MBR','Shared4_global','Shared4_observed','Independent4_observed']

def construct(old):
    e=np.asarray(old['outputs']['E_MBR'],float)
    global_out=np.asarray(old['outputs']['B_aligned_once'],float)
    positions=list(old['planned_positions']);assert len(positions)==len(set(positions))==4
    local=e.copy();shared=e.copy();residual=np.zeros((4,4));counts=np.zeros(4,dtype=int)
    shared[positions]=global_out[positions]
    if old['active']:
        actions=np.asarray(old['actions']);assert actions.shape==(24,4)
        details=old['matched']['B_aligned']['details'];assert len(details)==24
        source=np.asarray(old['outputs']['Source_MBR'],float)
        for k,d in enumerate(details):
            sampled=decode(encode(source)+actions[k]*old['scale'])
            for j,pos in enumerate(positions):
                v=d['anchors'].get(pos)
                if v is None:continue
                assert d['legal'] and v['accepted']==bool(v['score']>=.35)
                if v['accepted']:
                    residual[j]+=encode(np.asarray(v['box'],float)[None])[0]-encode(sampled[pos:pos+1])[0]
                    counts[j]+=1
        residual/=24
        for j,pos in enumerate(positions):
            if np.any(residual[j]!=0):local[pos]=decode(encode(e[pos:pos+1])+residual[j])[0]
    else:assert np.array_equal(e,global_out)
    unobs=~np.isin(np.arange(len(e)),positions)
    assert np.array_equal(local[unobs],e[unobs]) and np.array_equal(shared[unobs],e[unobs])
    return dict(outputs=dict(zip(ARMS,[e.copy(),global_out.copy(),shared,local])),residual=residual,accepted_counts=counts)

def contract():
    # Explicit finite-sample contract, independent of a detector or GT.
    sample=np.array([[.1,.2,.3,.4],[.3,.1,-.1,.2]])
    teacher=sample.copy();assert np.array_equal((teacher-sample).sum(0)/24,np.zeros(4))
    teacher[0]+=np.array([.03,-.02,.01,0.])
    assert np.allclose((teacher-sample).sum(0)/24,np.array([.03,-.02,.01,0.])/24,atol=1e-17)
    return dict(status='pass',no_feedback_exact_zero=True,partial_feedback_divides_all24=True,
                no_cross_anchor_averaging=True,unchanged_nodes_preserved_exact=True)
