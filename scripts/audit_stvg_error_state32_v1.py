"""Independent sealed-output numerical audit; DINO phase uses no GT or actionness."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
D=ROOT/'artifacts/stvg_error_state32_v1'
OLD=ROOT/'artifacts/stvg_dino_rollback32_v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sealed(p):
    assert sha(p)==read(Path(p).with_suffix('.receipt.json'))['sha256']
    return torch.load(p,map_location='cpu',weights_only=False)
def chart(b):
    b=np.asarray(b,dtype=np.float64)
    assert np.isfinite(b).all()and(b[:,2:]>0).all()
    return np.concatenate((b[:,:2],np.log(b[:,2:])),axis=1)

def correction_check(base,positions,target,saved):
    b=np.asarray(base);p=np.asarray(positions,dtype=int);t=np.asarray(target).reshape(-1,4)
    z=chart(b);other=np.setdiff1d(np.arange(len(b)),p)
    assert len(p)==len(t)==saved['reference_count']and len(set(p))==len(p)
    assert np.array_equal(saved['local'][other],b[other])
    assert np.array_equal(saved['local'][p],t)
    if not len(p):
        assert np.array_equal(saved['local'],b)and np.array_equal(saved['global_error'],b)
        assert np.array_equal(saved['delta'],np.zeros(4))
        return dict(references=0,exact_noop=True,max_error=0.)
    residual=chart(t)-z[p]
    # Solve four independent overdetermined constant fits, rather than calling correct().
    delta=np.linalg.lstsq(np.ones((len(p),1)),residual,rcond=None)[0][0]
    expected=np.column_stack((b[:,:2]+delta[:2],b[:,2:]*np.exp(delta[2:])))
    error=max(float(np.max(abs(delta-saved['delta']))),float(np.max(abs(expected-saved['global_error']))))
    assert error<2e-12
    before=float(np.sum(residual**2));after=float(np.sum((residual-delta)**2))
    assert abs(before-saved['before_z_SSE'])<1e-10*(1+before)
    assert abs(after-saved['after_z_SSE'])<1e-10*(1+after)
    assert abs(before-after-len(p)*np.dot(delta,delta))<1e-10*(1+before)
    assert np.max(abs(np.mean(chart(t)-chart(saved['global_error'])[p],axis=0)))<1e-12
    assert np.max(abs(np.diff(saved['global_error'][:,:2],axis=0)-np.diff(b[:,:2],axis=0)))<1e-12
    assert np.max(abs(np.diff(chart(saved['global_error'])[:,2:],axis=0)-np.diff(z[:,2:],axis=0)))<1e-12
    assert np.max(abs(saved['global_error'][:,2:]/b[:,2:]-np.exp(delta[2:])))<1e-12
    return dict(references=len(p),exact_noop=False,max_error=error)

def main(oracle=False):
    torch.set_num_threads(1);start=time.perf_counter();lock=read(D/'RUNTIME_LOCK.json')
    for f,h in {**lock['pins'],**lock['registries']}.items():assert sha(ROOT/f)==h
    bar=read(D/'DINO_FULL_BARRIER.json');assert bar['count']==32 and bar['GT_read']is False
    if oracle:
        assert read(D/'CPU_COMPLETION.json')['status']=='complete'
        allbar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
        assert allbar['DINO_full_barrier_sha256']==sha(D/'DINO_FULL_BARRIER.json')
    from scripts.audit_stvg_posterior_decision_tree_v1 import direct
    from vg_tta.stvg_posterior_decision_tree_v1 import expected_iou_fast
    details=[];candidate_checks=correction_checks=0;maxq=maxu=maxc=0.
    for f in bar['files']:
        assert sha(ROOT/f['path'])==f['sha256'];z=sealed(ROOT/f['path']);alias=z['alias']
        oldpath=OLD/'predictions'/(alias+'.pt');assert sha(oldpath)==lock['inputs'][str(oldpath.relative_to(ROOT))]
        original=sealed(oldpath);ids=np.asarray(z['frame_ids']);n=len(ids);index={int(v):i for i,v in enumerate(ids)}
        assert np.issubdtype(ids.dtype,np.integer)and(np.diff(ids)>=1).all()
        tables=[];native=[]
        # Only the Source prior logp0 and its legal index list are accessed.
        # raw_logits/actions/standardized actionness/target scores are never used.
        for off,o in enumerate(original['projection']['offsets']):
            pairs=o['ij'].numpy();lp=o['logp0'].numpy();localids=ids[off::2]
            assert np.array_equal(pairs,np.asarray(np.triu_indices(len(localids),1)))
            p=np.exp(lp-lp.max());p/=p.sum();table=np.zeros((n,n))
            a=np.array([index[int(v)]for v in localids[pairs[0]]]);b=np.array([index[int(v)]for v in localids[pairs[1]]])
            table[a,b]=p;tables.append(table)
            j=int(np.argmax(lp));native.append([int(localids[pairs[0,j]]),int(localids[pairs[1,j]])+1])
        assert len(tables)==2
        H=[t[::-1,:].cumsum(0)[::-1,:].cumsum(1)for t in tables]
        joint=np.pad(H[0]*H[1],((0,1),(1,0)))
        q=joint[:-1,1:]-joint[1:,1:]-joint[:-1,:-1]+joint[1:,:-1]
        qerror=float(np.max(abs(q-z['q'])));assert qerror<1e-12;maxq=max(maxq,qerror)
        assert abs(z['q'].sum()-1)<1e-12 and np.min(z['q'])>=0 and not np.tril(z['q']).any()
        assert z['intervals']['Native']==[min(v[0]for v in native),max(v[1]for v in native)]
        a,b,utility=direct(ids,z['q']);fast=expected_iou_fast(ids,z['q'])[a,b]
        uerror=float(np.max(abs(fast-utility)));assert uerror<1e-11;maxu=max(maxu,uerror)
        best=int(np.argmax(utility));assert z['intervals']['final_MBR']==[int(ids[a[best]]),int(ids[b[best]])+1]
        j=int(np.argmax(z['q']));s,e=divmod(j,n);assert z['intervals']['marginal_MAP']==[int(ids[s]),int(ids[e])+1]
        candidate_checks+=len(a)
        refs={'DINO_full':dict(positions=z['reference_positions'],targets=z['reference_targets'])}
        corrections={'DINO_full':z['corrections']}
        if oracle:
            o=sealed(D/'oracle'/(alias+'.pt'));sparse=sealed(D/'oracle_sparse_inputs'/(alias+'.pt'))
            for name,r in o['references'].items():
                assert r['positions']==sparse['references'][name]['positions']
                assert np.array_equal(r['targets'],sparse['references'][name]['targets'])
            assert o['unobserved_GT_used_for_predictions']is False
            refs.update(o['references']);corrections.update(o['corrections'])
        checks={}
        for ref,reference in refs.items():
            checks[ref]={}
            for base,bbox in z['bases'].items():
                c=correction_check(bbox,reference['positions'],reference['targets'],corrections[ref][base])
                maxc=max(maxc,c['max_error']);correction_checks+=1;checks[ref][base]=c
        details.append(dict(alias=alias,dataset=z['dataset'],candidate_count=len(a),
                            q_CDF_max_error=qerror,utility_max_error=uerror,corrections=checks))
        del original,z
    assert not torch.cuda.is_initialized()
    receipt=dict(status='pass',count=32,phase='all_sealed_sparse_reference_outputs'if oracle else'DINO_full_no_GT',
        GT_coordinates_read=oracle,unobserved_GT_read=False,actionness_values_read=False,source_logp0_only=True,
        exhaustive_candidate_checks=candidate_checks,correction_checks=correction_checks,
        max_q_CDF_error=maxq,max_utility_error=maxu,max_constant_LS_error=maxc,
        local_unobserved_nodes_exact=True,empty_reference_exact_noop=True,
        center_and_logwidth_increments_preserved=True,all_legal_zero_mass_candidates_included=True,
        DINO_barrier_GT_false=True,DINO_barrier_sha256=sha(D/'DINO_FULL_BARRIER.json'),
        original_pins_and_registries_exact=True,new_models_DINO_training_Student_updates=0,
        CPU_wall_seconds=time.perf_counter()-start,script_sha256=sha(__file__),files=details)
    dest=D/('ROOT_ALL_REFERENCE_NUMERICAL_AUDIT.json'if oracle else'ROOT_DINO_NUMERICAL_AUDIT_NO_GT.json')
    assert not dest.exists();dest.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items()if k!='files'}))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--oracle',action='store_true');main(p.parse_args().oracle)
