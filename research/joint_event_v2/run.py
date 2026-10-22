"""Structural null correction after preregistered v1 expanded all22 acceptances."""
import os,sys,time,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,OLD,B
D=R/'outputs/joint_event_hc64_v2';P=R/'results/joint_event_hc64_v2';V1=R/'outputs/joint_event_hc64_v1'
ARMS=['NativeG6','ExcessGuard','Excess','ShiftExcessGuard','RawExcessGuard']
def verify():
    p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
    assert sha(R/'outputs/tastvg_paper_v1/plan.json')==p['paper_plan_sha256']
    assert sha(V1/'plan.json')==p['v1_plan_sha256']and sha(V1/'sealed.json')==p['v1_seal_sha256']
    for k,v in p['code'].items():assert sha(R/k)==v,k
    for k,v in p['paper_code'].items():assert sha(R/k)==v,k
    for k,v in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/k)==v,k
    return p
def prepare():
    assert not(D/'plan.json').exists();v=read(V1/'plan.json');s=read(V1/'sealed.json');paper=read(R/'outputs/tastvg_paper_v1/plan.json')
    assert s['count']==64 and not s['GT_read']and read(V1/'state.json')['status']=='completed_scored'
    code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
    for k in ['research/joint_event_v1/core.py','research/joint_event_v1/run.py','research/boundary_feedback_v1/core.py','research/boundary_feedback_v1/score.py']:code[k]=sha(R/k)
    p=dict(schema='joint_event_hc64_v2',primary='ExcessGuard',arms=ARMS,queries=64,
        rationale='V1 all22 guarded changes expanded; score/log-duration correlation.9353. Remove mathematically present extent preference, not scan strength or duration parameters.',
        fixed=dict(feedback='v1 cached same target-identity-relative predicate margins and ridge1 projections, zero refitting',
            score='mean_k [log softIoU(mask_k,interval)-log softIoU(uniform_same_total_mask_k,interval)]',
            controls='Excess without guard; ShiftExcessGuard circular logit shift floor(T/4) and restore same physical occupancy, same guard; RawExcessGuard old event feedback, same correction/guard',
            unchanged='same q0, native anchor-class mass preservation, original MBR, all18-view/LOO positive gain and per-predicate gate',
            null_invariants='constant mask gives zero score for every interval; full-video interval gives zero score for any mask'),
        exposure='same64 already repeatedly exposed development; v2 chosen from v1 mechanism audit, not independent confirmation',
        no_GT_in_predictions=True,all64_new_predictions_sealed_before_new_score=True,query_updates=0,new_visual_forwards=0,new_decoder=0,new_expert=0,new_projection_fits=0,
        v1_plan_sha256=sha(V1/'plan.json'),v1_seal_sha256=sha(V1/'sealed.json'),prior_files=s['files'],bindings=v['bindings'],
        code=code,paper_code=paper['code'],provider_sources=paper['provider_sources'],paper_plan_sha256=sha(R/'outputs/tastvg_paper_v1/plan.json'),labels_sha256=v['labels_sha256'],
        maximum_CPU_seconds=600,full_status='paused_by_user',automatic_resume=False,spatial='unchanged original complete-box geometry and per-interval exactDP; fixed-native-space control',
        references=['https://minghangz.github.io/uploads/CPL/CPL_paper.pdf','https://github.com/minghangz/cpl/blob/main/models/cpl.py'])
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p)
    put(P/'protocol.json',{k:x for k,x in p.items()if k not in ['prior_files','bindings','paper_code','provider_sources']})
    for k in code:
        f=D/'source_snapshot'/k;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((R/k).read_bytes())
    put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
    import numpy as np,torch
    from stvg.runtime import guard as annotation_guard
    from stvg.temporal import final_mbr
    from stvg.spatial import SpatialGeometry,evidence_dp
    from research.joint_event_v1.core import physical_membership,calibrated_masks,guard
    from research.joint_event_v2.core import excess_scores
    from research.boundary_feedback_v1.core import reweight
    p=verify();torch.set_num_threads(1);sys.addaudithook(annotation_guard);assert not(D/'sealed.json').exists()and not torch.cuda.is_initialized()
    start=time.perf_counter();files={};put(D/'state.json',dict(status='predicting_CPU',fingerprint=p['fingerprint']))
    for row in records():
        k=row['id'];e=p['prior_files'][k];binding=p['bindings'][k]
        assert sha(V1/e['path'])==e['sha256']and sha(V1/e['trace_path'])==e['trace_sha256']and sha(B/'captures'/f'{k}.pt')==binding['capture_sha256']
        tr=torch.load(V1/e['trace_path'],map_location='cpu',weights_only=False);cap=torch.load(B/'captures'/f'{k}.pt',map_location='cpu',weights_only=False)
        ids=cap['frame_ids'];pairs=tr['pairs'];group=tr['groups'];q0=tr['q']['NativeG6'];overlap,width,lengths=physical_membership(ids,pairs)
        base=final_mbr(ids,q0);base_idx=int(np.flatnonzero((pairs==base['indices']).all(1))[0]);tick=time.perf_counter();qs={'NativeG6':q0};iv={'NativeG6':base['interval']};details={};saved={}
        for kind in ['Relative','Raw','Shift']:
            entry=tr['semantic'].get('Raw'if kind=='Raw'else'Relative');names=['Excess','ExcessGuard']if kind=='Relative'else ['RawExcessGuard']if kind=='Raw'else ['ShiftExcessGuard']
            if entry is None:
                for a in names:qs[a]=q0;iv[a]=base['interval'];details[a]=dict(accept=False,reason='no_valid_semantic_contrast')
                continue
            masks=entry['masks'];skills=entry['skills']
            if kind=='Shift':
                shifted=[[np.roll(np.log(v/np.maximum(1-v,1e-30)),len(ids)//4,axis=0)for v in views]for views in masks]
                masks=calibrated_masks(shifted,width,tr['occupancy'])
            combined=(masks[0][0]+masks[1][0])/2;score,logs=excess_scores(combined,overlap,width,lengths)
            view_scores=[excess_scores(v,overlap,width,lengths)[0]for view in masks for v in view]
            q=reweight(q0,pairs,score,group);decision=final_mbr(ids,q);idx=int(np.flatnonzero((pairs==decision['indices']).all(1))[0]);accept,diag=guard(idx,base_idx,view_scores,logs,skills)
            if kind=='Relative':
                qs['Excess']=q;iv['Excess']=decision['interval'];details['Excess']=dict(guard_would_accept=accept,**diag)
                qs['ExcessGuard']=q if accept else q0;iv['ExcessGuard']=decision['interval']if accept else base['interval'];details['ExcessGuard']=diag
            else:
                a=names[0];qs[a]=q if accept else q0;iv[a]=decision['interval']if accept else base['interval'];details[a]=diag
            saved[kind]=dict(score=score,skills=skills,combined=combined)
        feedback_seconds=time.perf_counter()-tick;geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={}
        for a in ARMS:arrays[a]=evidence_dp(geom.for_interval(iv[a]))['final'].numpy()
        with np.load(V1/e['path'],allow_pickle=False)as z:
            assert np.array_equal(arrays['NativeG6'],z['NativeG6'])and np.array_equal(iv['NativeG6'],z['NativeG6_interval'])
            arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
        out=D/'predictions'/f'{k}.npz';assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,**arrays,**{a+'_interval':s for a,s in iv.items()},frame_ids=ids,positions8=cap['planned_positions'])
        trace=D/'traces'/f'{k}.pt';trace.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(q=qs,details=details,semantic=saved,text_metadata=tr['text_metadata'],pairs=pairs,groups=group,GT_read=False),trace)
        files[k]=dict(path=str(out.relative_to(D)),sha256=sha(out),trace_path=str(trace.relative_to(D)),trace_sha256=sha(trace),feedback_seconds=feedback_seconds,seconds=time.perf_counter()-tick)
        put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start));print(k,{a:bool(details[a].get('accept'))for a in ['ExcessGuard','ShiftExcessGuard','RawExcessGuard']},flush=True)
        assert time.perf_counter()-start<p['maximum_CPU_seconds']
    verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
if __name__=='__main__':globals()[sys.argv[1]]()
