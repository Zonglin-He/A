"""Independent NumPy geometry, occupancy, sampling and actual query reconstruction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def occ(q):
    o=np.zeros(len(q))
    for i,j in zip(*np.triu_indices(len(q),1)):o[i:j+1]+=q[i,j]
    return o

def main():
    D=R/'artifacts/stvg_temporal_event_query128_v1';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';p=read(D/'PLAN_LOCK.json');b=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert b['count']==128;query_checks=0;area_cells=0;noops=0;maxq=maxa=maxo=maxr=max_relative=0.;changed=0
    for r,f in zip(p['records'],b['files']):
        assert sha(R/f['path'])==f['sha256'];z=load(R/f['path']);c=load(MEM/'temporal_inputs'/f"{r['alias']}.pt");o0=occ(z['q0']);o1=occ(z['q1']);err=max(abs(o0-z['occupancy0']).max(),abs(o1-z['occupancy1']).max());assert err<1e-12;maxo=max(maxo,float(err));changed+=z['outputs']['Student1']['interval']!=z['outputs']['Native']['interval']
        ii,jj=np.triu_indices(65,1);u=np.random.default_rng(np.random.SeedSequence([41,r['query_ordinal']])).random(4);assert np.array_equal(u,z['paired_uniforms'])
        for si,q in enumerate([z['q0'],z['q1']]):
            cumulative=np.cumsum(q[ii,jj]);cumulative[-1]=1.;k=np.searchsorted(cumulative,u,side='right');assert np.array_equal(np.stack([ii[k],jj[k]],-1),z['sampled_pairs'][si])
        tw=dict(GAll=np.ones(65),RAll=np.ones(65),GOcc0=o0,ROcc0=o0,ROcc1=o1)
        for si,pairs in enumerate(z['sampled_pairs']):
            for k,(i,j)in enumerate(pairs):tw[f'Sample{si}_{k}']=((np.arange(65)>=i)&(np.arange(65)<=j)).astype(float)
        for v,tr in enumerate(z['query_traces']):
            g=c['grids'][v];h,w=g['h'],g['w'];l=h*w;H=c['inputs'][v]['encoded_feature'][-l:].permute(1,0,2).numpy().astype('float64');a=tr['attention']['global_attention'].numpy().astype('float64');ra=tr['attention']['refer_attention'].numpy().astype('float64');coverage=tr['attention']['coverage'].numpy();native=tr['native_query'].numpy().astype('float64');native_norm=np.linalg.norm(native)
            for t,box in enumerate(z['conditioning_boxes'][v::2]):
                valid=g['valid'][t].numpy().reshape(h,w);vh=int(valid.any(1).sum());vw=int(valid.any(0).sum());lo=box[:2]-box[2:]/2;hi=box[:2]+box[2:]/2;xx=np.arange(vw);yy=np.arange(vh);dx=np.maximum(0,np.minimum(1,np.minimum(hi[0]*vw,xx+1)-np.maximum(lo[0]*vw,xx)));dy=np.maximum(0,np.minimum(1,np.minimum(hi[1]*vh,yy+1)-np.maximum(lo[1]*vh,yy)));area=np.zeros((h,w));area[:vh,:vw]=dy[:,None]*dx;err=float(abs(area.ravel()-coverage[t]).max());assert err<1e-7;maxa=max(maxa,err);area_cells+=h*w
                raw=a[t]*area.ravel();expected=a[t]if raw.sum()<=0 else raw*a[t].sum()/raw.sum();e=float(abs(expected-ra[t]).max());scale=max(1.,float(abs(ra[t]).max()));assert e/scale<16*np.finfo(np.float32).eps;assert abs(expected.sum()-ra[t].sum())/max(1.,abs(expected.sum()))<16*np.finfo(np.float32).eps;maxr=max(maxr,e);max_relative=max(max_relative,e/scale)
            for name,qmeta in tr['queries'].items():
                if qmeta['noop']:noops+=1;continue
                ww=tw[name][v::2];assert float(abs(ww-qmeta['time_weights'].numpy()).max())<1e-12
                attention=a if name.startswith('G')else ra;weights=attention*ww[:,None];raw=(H*weights[...,None]).sum((0,1))/weights.sum();q=(raw*(native_norm/np.linalg.norm(raw))).astype('float32');err=float(abs(q-qmeta['query'].numpy()).max());assert err<1e-6;maxq=max(maxq,err);query_checks+=1
    assert query_checks+noops==3328 and changed==113
    cost=read(D/'COST.json');assert cost['conditioned_temporal_offset_decodes']==3328 and cost['native_temporal_checks']==256
    assert all(sha(R/f)==h for f,h in p['pins'].items())
    out=dict(status='pass',query_numpy_reconstructions=query_checks,noops=noops,independent_fractional_cells=area_cells,independent_policy_samples=1024,actual_student_changed_parents=changed,max_query_error=maxq,max_cell_area_error=maxa,max_occupancy_error=maxo,max_raw_attention_error=maxr,max_scaled_attention_error=max_relative,float32_attention_relative_bound=16*np.finfo(np.float32).eps,audit_precision_revision_disclosed=True,new_GPU=0,new_expert=0,new_training=0)
    put(R/'results/stvg_error_state/2026-10-20/temporal_event_query128/QUERY_AUDIT_PUBLIC.json',out);print(out)

if __name__=='__main__':main()
