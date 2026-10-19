"""Independent CPU readback; no new inference or candidate choice."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import itertools,time
import numpy as np
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_observation_factorial32_v1 import D,P,PRE,ARMS,verify

def chart(b):
    b=np.asarray(b,dtype=float);return np.concatenate((b[...,:2],np.log(b[...,2:])),axis=-1)

def independent_E(pools,source,scale,cutoff):
    frames=[pools[k]for k in sorted(pools)]
    if not frames:return source.copy(),[]
    values=[chart(f['boxes'])-chart(source)[f['position']]for f in frames]
    terms=[]
    for f in frames:
        score=np.maximum(np.asarray(f['scores'],float),1e-12);terms.append((np.log(score.max())-np.log(score))*cutoff/16)
    candidates=[]
    for assignment in itertools.product(*[range(-1,len(v))for v in values]):
        keep=[v[k]for v,k in zip(values,assignment)if k!=-1]
        delta=np.mean(keep,axis=0)if keep else np.zeros(4);losses=[np.sum(((v-delta)/scale)**2,axis=1)+c for v,c in zip(values,terms)]
        choice=tuple(int(np.argmin(v))if v.min()<cutoff else -1 for v in losses)
        candidates.append((sum(min(cutoff,float(v.min()))for v in losses),choice,assignment,delta))
    chosen=min(candidates,key=lambda z:(z[0],z[1],z[2]));delta=chosen[3]
    if np.array_equal(delta,np.zeros(4)):return source.copy(),list(chosen[1])
    c=chart(source)+delta;return np.concatenate((c[:,:2],np.exp(c[:,2:])),axis=1),list(chosen[1])

def nms3(det):
    b=np.asarray(det['all_boxes'],np.float32);scores=np.asarray(det['all_phrase_scores'],np.float32)
    xy=np.clip(np.concatenate((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1),0,1)
    order=np.argsort(-scores,kind='stable');selected=[]
    while len(order)and len(selected)<3:
        i=int(order[0]);selected.append(i);rest=order[1:]
        lo=np.maximum(xy[i,:2],xy[rest,:2]);hi=np.minimum(xy[i,2:],xy[rest,2:]);inter=np.maximum(hi-lo,0).prod(1)
        area=np.maximum(xy[:,2:]-xy[:,:2],0).prod(1);union=area[i]+area[rest]-inter
        overlap=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
        order=rest[overlap<=.5]
    selected=[i for i in selected if (xy[i,2:]>xy[i,:2]).all()]
    local=np.concatenate(((xy[selected,:2]+xy[selected,2:])/2,xy[selected,2:]-xy[selected,:2]),1)
    return selected,local

def main():
    import torch
    torch.set_num_threads(4);started=time.perf_counter();plan=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['count']==32 and not torch.cuda.is_initialized()
    result=read(P/'RESULTS_PUBLIC.json');assert result['OPD_not_run'] and result['count']==32
    rr={r['alias']:r for r in result['rows']};old={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-19/e_retaining1792_route32/RESULTS_PUBLIC.json')['rows']if r['stage']=='real'}
    cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains']
    counters=dict(crop_geometry=0,NMS_candidates=0,E_outputs=0,CDF_rectangles=0,MBR_utility_checks=0,baseline_metrics=0,bootstrap_endpoints=0)
    maximum=dict(crop_error=0.,E_error=0.,CDF_error=0.,utility_error=0.)
    for r in plan['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'));s=load(PRE/'source'/(r['alias']+'.pt'));e=load(PRE/'E'/(r['alias']+'.pt'))
        rawrow=read(ROOT/f'artifacts/decota_paper_experiments_v1/{r["dataset"]}/PLAN.json')['rows'][r['query_ordinal']]
        W,H=rawrow['input']['width'],rawrow['input']['height']
        source=np.asarray(s['Source'],float);pool={}
        for a in s['expert']['anchors']['single4']:
            p=s['expert']['observations'][('original',a['position'])]['probe']
            pool[a['position']]=dict(position=a['position'],boxes=np.asarray(p['boxes'],float),scores=np.asarray(p['target_scores'],float))
        for ob in z['spatial']['observations']:
            p=ob['position'];box=e['E'][p];lo=np.floor((box[:2]-box[2:])*[W,H]).astype(int);hi=np.ceil((box[:2]+box[2:])*[W,H]).astype(int)
            bounds=[int(max(0,lo[0])),int(max(0,lo[1])),int(min(W,hi[0])),int(min(H,hi[1]))]
            assert bounds==ob['bounds'];counters['crop_geometry']+=1
            if not ob['called']:continue
            probe=ob['probe'];local=np.asarray(ob['local_boxes'],float);det=ob['detector']
            if 'target_all_scores'in det:
                indices,native=nms3(det);assert indices==probe['candidate_ids'];assert np.allclose(native,local,rtol=0,atol=1e-7)
                raw=np.asarray(det['raw_token_logits'],float);token=np.asarray(det['entity_tokens']);target=(1/(1+np.exp(-raw[:,token]))).mean(1)
                assert np.allclose(target[indices],probe['target_scores'],rtol=0,atol=2e-7);counters['NMS_candidates']+=len(indices)
            x,y,right,down=bounds;answer=np.zeros_like(local)
            answer[:,0]=(local[:,0]*(right-x)+x)/W;answer[:,1]=(local[:,1]*(down-y)+y)/H
            answer[:,2]=local[:,2]*(right-x)/W;answer[:,3]=local[:,3]*(down-y)/H
            err=float(np.abs(answer-np.asarray(probe['boxes'])).max())if len(local)else 0.;maximum['crop_error']=max(maximum['crop_error'],err);assert err<1e-12
            scores=np.asarray(probe['target_scores'],float);ix=np.argsort(-scores,kind='stable')
            margin=float(scores[ix[0]]-(scores[ix[1]]if len(ix)>1 else 0.))if len(ix)else 0.
            accept=bool(len(ix)and scores[ix[0]]>=.35 and margin>=.05);assert accept==probe['accepted']
            if accept:pool[p]=dict(position=p,boxes=answer,scores=scores)
        c=cal['vidstg'if r['dataset']=='hc2'else'hc2'];out,choices=independent_E(pool,source,np.asarray(c['scale']),c['truncation'])
        err=float(np.abs(out-z['outputs']['S_crop']).max());maximum['E_error']=max(maximum['E_error'],err)
        assert err<1e-12 and choices==z['spatial']['estimate']['selected'];counters['E_outputs']+=1
        assert np.array_equal(z['outputs']['ST'],z['outputs']['S_crop'])
        for arm in ['E_MBR','T_dense','T_gridonly']:assert np.array_equal(z['outputs'][arm],e['E'])
        ids=np.asarray(z['dense_frame_ids']);q=z['temporal']['posterior'];n=len(ids);assert (q>=0).all()and abs(q.sum()-1)<1e-12
        components=[]
        for rec,logit in zip(z['temporal']['records'],z['temporal']['logits']):
            zz=np.asarray(logit,float)[0];a,b=np.triu_indices(len(zz),1);v=zz[a,0]+zz[b,1];v=np.exp(v-v.max());v/=v.sum();fid=np.asarray(rec['frame_ids'])
            components.append((fid[a],fid[b],v))
        for a,b in [(0,n-1),(0,n//2),(n//3,n-1),(n//4,3*n//4),(n//2,n//2)]:
            exact=np.prod([float(p[(st>=ids[a])&(en<=ids[b])].sum())for st,en,p in components])
            actual=float(q[a:,:b+1].sum());err=abs(exact-actual);maximum['CDF_error']=max(maximum['CDF_error'],err);assert err<1e-11;counters['CDF_rectangles']+=1
        starts,ends=np.triu_indices(n,1);qa=q[starts,ends];fs=ids[starts];fe=ids[ends]+1
        u=z['temporal']['dense_mbr']['utility_grid'];checklist=[tuple(z['temporal']['dense_mbr']['indices'])]+[(int(starts[k]),int(ends[k]))for k in np.linspace(0,len(starts)-1,7,dtype=int)]
        for a,b in checklist:
            left,right=ids[a],ids[b]+1;inter=np.maximum(0,np.minimum(right,fe)-np.maximum(left,fs));utility=float((qa*inter/(right-left+fe-fs-inter)).sum())
            err=abs(utility-u[a,b]);maximum['utility_error']=max(maximum['utility_error'],err);assert err<1e-11;counters['MBR_utility_checks']+=1
        for key in ['v','t','s','observed_actual_s','unobserved_actual_s']:
            aa=rr[r['alias']]['values']['E_MBR'][key];bb=old[r['alias']]['values']['E_MBR'][key]
            assert (aa is None and bb is None)or abs(aa-bb)<1e-12;counters['baseline_metrics']+=1
    for ds,g in result['groups'].items():
        rows=[r for r in result['rows']if r['dataset']==ds]
        for pair,epvals in g['paired'].items():
            a,b=pair.split('_minus_')
            for ep,record in epvals.items():
                data=np.array([100*(r['values'][a][ep]-r['values'][b][ep])for r in rows if r['values'][a][ep]is not None and r['values'][b][ep]is not None]);assert len(data)==record['parents']
                if len(data):
                    idx=np.random.default_rng(20261018).integers(len(data),size=(10000,len(data)))
                    assert abs(data.mean()-record['mean'])<1e-12 and np.allclose(np.quantile(data[idx].mean(1),[.025,.975]),record['CI95'],rtol=0,atol=1e-12)
                counters['bootstrap_endpoints']+=1
    supplement=read(D/'DEPENDENCY_BINDING_SUPPLEMENT.json')
    for f in supplement['files']:assert sha(ROOT/f['path'])==f['sha256']and f['last_modified_before_plan']
    assert supplement['DINO_weight_actual_SHA256']==supplement['DINO_expected_SHA256']
    audit=dict(status='pass',scope='CPU geometry, independent NMS/score, exhaustive E, rectangle envelope CDF, direct sampled MBR utilities, old baseline and all paired statistics',
        count=32,counters=counters,max_errors=maximum,all32_before_GT=True,OPD_not_run=True,
        dependency_supplement_not_original_prepin=True,seconds=time.perf_counter()-started,new_GPU_or_inference=0)
    put(P/'ROOT_INDEPENDENT_AUDIT.json',audit);print(audit,flush=True)

if __name__=='__main__':main()
