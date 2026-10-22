"""Nine actual interval decisions, same corrected semantic feedback and guard."""
import os,sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
V1=R/'outputs/joint_event_hc64_v1';V2=R/'outputs/joint_event_hc64_v2'
D=R/'outputs/joint_event_hc64_v3';P=R/'results/joint_event_hc64_v3'
ARMS=['NativeG6','LocalGuard','Local','ShiftLocalGuard','RawLocalGuard']
def verify():
    p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
    assert sha(V1/'sealed.json')==p['v1_seal_sha256']and sha(V2/'sealed.json')==p['v2_seal_sha256']
    assert sha(R/'outputs/tastvg_paper_v1/plan.json')==p['paper_plan_sha256']
    for k,v in p['code'].items():assert sha(R/k)==v,k
    for k,v in p['paper_code'].items():assert sha(R/k)==v,k
    for k,v in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/k)==v,k
    return p
def prepare():
    assert not(D/'plan.json').exists();v=read(V2/'plan.json');paper=read(R/'outputs/tastvg_paper_v1/plan.json')
    code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
    for k in ['research/joint_event_v1/core.py','research/joint_event_v1/run.py','research/joint_event_v2/core.py','research/boundary_feedback_v1/score.py']:code[k]=sha(R/k)
    p=dict(schema='joint_event_hc64_v3',queries=64,arms=ARMS,primary='LocalGuard',
        rationale='V2 removes extent bias but modifies7/64 despite39 having predictive projections. Isolate whether posterior MBR masks usable local semantic corrections.',
        fixed=dict(candidate_set='native start/end each moved by at most1 original physical sampled node, valid s<e, identical eight-anchor membership; at most9 pairs',
            local='highest combined full-view excess whole-event score; ties prefer native then smallest total boundary movement',
            guard='same both-offset positive LOO skill and all18-view score improvement plus no predicate score loss; select maximum worst-view gain among feasible local pairs',
            shifted='same floor(T/4) logit profile shift and native occupancy restoration as v2',
            unchanged='all semantic/projector/mask/null correction/evidence/space unchanged; no reward scale, temperature or radius sweep'),
        prior_files=v['prior_files'],bindings=v['bindings'],v1_seal_sha256=sha(V1/'sealed.json'),v2_seal_sha256=sha(V2/'sealed.json'),v2_plan_sha256=sha(V2/'plan.json'),
        code=code,paper_code=paper['code'],provider_sources=paper['provider_sources'],paper_plan_sha256=sha(R/'outputs/tastvg_paper_v1/plan.json'),labels_sha256=v['labels_sha256'],
        maximum_CPU_seconds=600,GT_in_inference=False,all64_seal_before_new_GT=True,new_visual_forwards=0,new_expert=0,new_projection_fits=0,query_steps=0,
        exposure='same64 exposed iterative development; no independent efficacy claim or automatic method selection',full_status='paused_by_user',automatic_resume=False)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:x for k,x in p.items()if k not in ['prior_files','bindings','paper_code','provider_sources']})
    for k in code:
        f=D/'source_snapshot'/k;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((R/k).read_bytes())
    put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def select_local(pairs,groups,base_index,score,logs,views,skills):
    import numpy as np
    from research.joint_event_v1.core import guard
    base=pairs[base_index];allowed=np.flatnonzero((np.max(abs(pairs-base),axis=1)<=1)&(groups==groups[base_index]))
    def tie(j):return (int(np.abs(pairs[j]-base).sum()),int(pairs[j,0]),int(pairs[j,1]))
    raw=min(allowed,key=lambda j:(-float(score[j]),*tie(j)))
    valid=[];diagnostics={}
    for j in allowed:
        ok,diag=guard(j,base_index,views,logs,skills);diagnostics[int(j)]=diag
        if ok:valid.append(j)
    picked=min(valid,key=lambda j:(-diagnostics[int(j)]['minimum_view_gain'],*tie(j)))if valid else base_index
    diag=diagnostics[int(picked)];diag.update(local_candidates=int(len(allowed)),feasible_candidates=int(len(valid)),accept=bool(picked!=base_index))
    return int(raw),int(picked),diag
def predict():
    import torch,numpy as np
    from stvg.runtime import guard as annotation_guard
    from stvg.temporal import final_mbr
    from stvg.spatial import SpatialGeometry,evidence_dp
    from research.joint_event_v1.core import physical_membership,calibrated_masks
    from research.joint_event_v2.core import excess_scores
    p=verify();torch.set_num_threads(1);sys.addaudithook(annotation_guard);assert not(D/'sealed.json').exists()and not torch.cuda.is_initialized()
    start=time.perf_counter();files={};put(D/'state.json',dict(status='predicting_CPU',fingerprint=p['fingerprint']))
    for row in records():
        k=row['id'];e=p['prior_files'][k];binding=p['bindings'][k]
        assert sha(V1/e['path'])==e['sha256']and sha(V1/e['trace_path'])==e['trace_sha256']and sha(B/'captures'/f'{k}.pt')==binding['capture_sha256']
        tr=torch.load(V1/e['trace_path'],map_location='cpu',weights_only=False);cap=torch.load(B/'captures'/f'{k}.pt',map_location='cpu',weights_only=False)
        ids=cap['frame_ids'];pairs=tr['pairs'];q0=tr['q']['NativeG6'];overlap,width,lengths=physical_membership(ids,pairs);base=final_mbr(ids,q0);base_idx=int(np.flatnonzero((pairs==base['indices']).all(1))[0]);tick=time.perf_counter()
        iv={'NativeG6':base['interval']};details={};saved={};qs={'NativeG6':q0}
        for kind in ['Relative','Raw','Shift']:
            entry=tr['semantic'].get('Raw'if kind=='Raw'else'Relative');names=['Local','LocalGuard']if kind=='Relative'else ['RawLocalGuard']if kind=='Raw'else ['ShiftLocalGuard']
            if entry is None:
                for a in names:iv[a]=base['interval'];details[a]=dict(accept=False,reason='no_valid_semantic_contrast')
                continue
            masks=entry['masks'];skills=entry['skills']
            if kind=='Shift':
                logits=[[np.roll(np.log(v/np.maximum(1-v,1e-30)),len(ids)//4,axis=0)for v in views]for views in masks];masks=calibrated_masks(logits,width,tr['occupancy'])
            combined=(masks[0][0]+masks[1][0])/2;score,logs=excess_scores(combined,overlap,width,lengths);views=[excess_scores(v,overlap,width,lengths)[0]for vv in masks for v in vv]
            raw,pick,diag=select_local(pairs,tr['groups'],base_idx,score,logs,views,skills)
            def interval(j):return [int(ids[pairs[j,0]]),int(ids[pairs[j,1]])+1]
            if kind=='Relative':
                iv['Local']=interval(raw);details['Local']=dict(accept=bool(raw!=base_idx),guard_would_accept=bool(pick==raw and raw!=base_idx))
                iv['LocalGuard']=interval(pick);details['LocalGuard']=diag
            else:a=names[0];iv[a]=interval(pick);details[a]=diag
            saved[kind]=dict(score=score,skills=skills)
        feedback_seconds=time.perf_counter()-tick;geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
        with np.load(V1/e['path'],allow_pickle=False)as z:
            assert np.array_equal(arrays['NativeG6'],z['NativeG6'])and np.array_equal(iv['NativeG6'],z['NativeG6_interval']);arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
        out=D/'predictions'/f'{k}.npz';assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays,**{a+'_interval':s for a,s in iv.items()},frame_ids=ids,positions8=cap['planned_positions'])
        trace=D/'traces'/f'{k}.pt';trace.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,details=details,semantic=saved,text_metadata=tr['text_metadata'],pairs=pairs,GT_read=False),trace)
        files[k]=dict(path=str(out.relative_to(D)),sha256=sha(out),trace_path=str(trace.relative_to(D)),trace_sha256=sha(trace),feedback_seconds=feedback_seconds,seconds=time.perf_counter()-tick)
        put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start));print(k,bool(details['LocalGuard']['accept']),flush=True);assert time.perf_counter()-start<p['maximum_CPU_seconds']
    verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
if __name__=='__main__':globals()[sys.argv[1]]()
