"""One complete cache-only P9 audit; all test-visible signals seal before GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import collections,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts.stvg_opd_temporal_guided_p7_common_v1 import BASE as OLD,A,B,OUT as OLDOUT,verify as old_verify
BASE=ROOT/'artifacts/stvg_opd_evidence_actionability_audit_v1'
OUT=ROOT/'results/stvg_opd_evidence_actionability_audit/2026-10-12'
META=['public_cell','parent_id','dataset','order','previous_P6']
SIGNALS=['holdout_agreement_gain','detector_score','before_training_agreement','training_agreement_gain','teacher_geometric_consistency','negative_output_displacement']
EPS=1e-12;SEED=20261012;BOOT=10000

def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]
def put_rows(path,rr):
    assert not path.exists()
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:
            for row in rr:f.write((json.dumps(row,allow_nan=False,separators=(',',':'))+'\n').encode())
def mean(x):return float(np.mean(x)) if len(x) else None
def xyxy(boxes,width=1.,height=1.,clip=False):
    b=np.asarray(boxes,dtype=np.float64).reshape(-1,4);out=np.empty_like(b)
    out[:,:2]=b[:,:2]-b[:,2:]/2;out[:,2:]=b[:,:2]+b[:,2:]/2
    out*=np.array([width,height,width,height]);return np.maximum(out,0) if clip else out
def iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    wh=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0);inter=wh[...,0]*wh[...,1]
    wa=np.maximum(a[...,2:]-a[...,:2],0);wb=np.maximum(b[...,2:]-b[...,:2],0)
    den=wa[...,0]*wa[...,1]+wb[...,0]*wb[...,1]-inter
    return np.divide(inter,den,out=np.zeros_like(inter),where=den>0)
def agreement(boxes,anchors,clip):
    return mean([float(iou(xyxy([boxes[int(a['position'])]],clip=clip)[0],xyxy([a['box']],clip=clip)[0])) for a in anchors])
def visible(before,after,training,other,visited,clip):
    hold=[a for a in other if int(a['position']) not in set(visited)]
    tb=agreement(before,training,clip);ta=agreement(after,training,clip)
    hb=agreement(before,hold,clip);ha=agreement(after,hold,clip)
    pairs=[float(iou(xyxy([a['box']],clip=clip)[0],xyxy([b['box']],clip=clip)[0])) for j,a in enumerate(training) for b in training[j+1:]]
    return dict(holdout_anchors=len(hold),training_anchors=len(training),holdout_before_agreement=hb,holdout_after_agreement=ha,
        holdout_agreement_gain=None if hb is None else ha-hb,detector_score=mean([float(a['score']) for a in training]),
        before_training_agreement=tb,training_agreement_gain=None if tb is None else ta-tb,
        teacher_geometric_consistency=mean(pairs),negative_output_displacement=-float(np.linalg.norm(after-before,axis=1).mean()))

def rank_distribution(scores,labels,weights):
    """Exact tie-group ROC/AP under arbitrary nonnegative bootstrap counts."""
    s=np.asarray(scores,float);y=np.asarray(labels,bool);w=np.atleast_2d(weights).astype(float)
    order=np.argsort(-s,kind='stable');s=s[order];y=y[order];w=w[:,order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    p=np.add.reduceat(w*y,starts,axis=1);n=np.add.reduceat(w*(~y),starts,axis=1)
    cp=np.cumsum(p,axis=1);cn=np.cumsum(n,axis=1);P=p.sum(1);N=n.sum(1)
    valid=(P>0)&(N>0);roc=np.full(len(w),np.nan);ap=roc.copy();prev=roc.copy()
    roc[valid]=(p*(N[:,None]-cn+.5*n)).sum(1)[valid]/(P[valid]*N[valid])
    precision=np.divide(cp,cp+cn,out=np.zeros_like(cp),where=(cp+cn)>0)
    ap[valid]=(p*precision).sum(1)[valid]/P[valid];prev[valid]=P[valid]/(P[valid]+N[valid])
    return roc,ap,prev
def summary_distribution(values):
    x=np.asarray(values,float);valid=x[np.isfinite(x)]
    return dict(ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,valid_resamples=len(valid),undefined_resamples=len(x)-len(valid))
def rank_stats(records,score,outcome):
    eligible=[r for r in records if r.get(score) is not None and r.get(outcome) is not None and abs(r[outcome])>EPS]
    n=len(eligible);p=sum(r[outcome]>EPS for r in eligible);neg=n-p
    base=dict(total_parents=len(records),eligible_parents=n,positive_parents=p,negative_parents=neg,positive_prevalence=p/n if n else None,
        signal_undefined_parents=sum(r.get(score) is None for r in records),neutral_or_undefined_outcome_parents=sum(r.get(outcome) is None or abs(r[outcome])<=EPS for r in records))
    if not p or not neg:return dict(**base,AUROC=None,AP=None,AUROC_bootstrap=None,AP_bootstrap=None)
    scores=[r[score] for r in eligible];labels=[r[outcome]>EPS for r in eligible]
    point=rank_distribution(scores,labels,np.ones((1,n)))
    idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n));weights=np.zeros((BOOT,n),np.int16)
    np.add.at(weights,(np.repeat(np.arange(BOOT),n),idx.ravel()),1)
    roc,ap,prev=rank_distribution(scores,labels,weights)
    return dict(**base,AUROC=float(point[0][0]),AP=float(point[1][0]),AUROC_bootstrap=summary_distribution(roc),AP_bootstrap=summary_distribution(ap))
def matched_rank(records,reference,outcome):
    rr=[r for r in records if r.get('holdout_agreement_gain') is not None and r.get(reference) is not None and r.get(outcome) is not None and abs(r[outcome])>EPS]
    n=len(rr);p=sum(r[outcome]>EPS for r in rr);d=dict(eligible_parents=n,positive_parents=p,negative_parents=n-p)
    if not p or p==n:return dict(**d,AUROC_difference=None,ci95=None,valid_resamples=0,undefined_resamples=BOOT)
    idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n));w=np.zeros((BOOT,n),np.int16);np.add.at(w,(np.repeat(np.arange(BOOT),n),idx.ravel()),1)
    y=[r[outcome]>EPS for r in rr];one=np.ones((1,n))
    a=rank_distribution([r['holdout_agreement_gain'] for r in rr],y,w)[0];b=rank_distribution([r[reference] for r in rr],y,w)[0]
    point=rank_distribution([r['holdout_agreement_gain'] for r in rr],y,one)[0][0]-rank_distribution([r[reference] for r in rr],y,one)[0][0]
    return dict(**d,AUROC_difference=float(point),**summary_distribution(a-b))
def parent_records(rr):
    by=collections.defaultdict(list)
    for r in rr:by[r['parent_id']].append(r)
    out=[]
    for parent,group in by.items():
        z=dict(parent_id=parent)
        for k in group[0]:
            if k in META or k=='scope':continue
            values=[r[k] for r in group if r[k] is not None]
            if values and all(isinstance(v,(bool,int,float)) for v in values):z[k]=math.fsum(values)/len(values)
            else:z[k]=None
        z['contributing_orders']=len(group);out.append(z)
    return out
def cross_table(rr,outcome,strict=False):
    c=collections.Counter();teacher_unknown=0;known=0
    for r in rr:
        q=r.get('teacher_quality')
        if q is None:teacher_unknown+=1;continue
        known+=1;correct=r.get('teacher_fully_labeled_correct')==1 if strict else q>=.5
        v=r.get(outcome);label='undefined' if v is None else 'positive' if v>EPS else 'negative' if v< -EPS else 'neutral'
        c[('geometrically_correct' if correct else 'not_geometrically_correct')+'_'+label]+=1
    return dict(total_parents=len(rr),teacher_unknown_parents=teacher_unknown,teacher_labeled_parents=known,cells=dict(c))
def aggregate(rr):
    result={}
    for ds in ['hc2','vidstg']:
        result[ds]={}
        for scope in ['Omega0','Omega1']:
            result[ds][scope]={}
            for subset in ['all128','previous_P6_32','added96']:
                result[ds][scope][subset]={}
                for order in ['both','order1','order2']:
                    chosen=[r for r in rr if r['dataset']==ds and r['scope']==scope and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32')) and (order=='both' or r['order']==order)]
                    parents=parent_records(chosen);block=dict(update_cells=len(chosen),parents=len(parents),rank={},matched={},cross={},mean={},outcomes={})
                    for outcome in ['common_U_delta','own_U_delta']:
                        vals=[r[outcome] for r in parents if r[outcome] is not None]
                        block['outcomes'][outcome]=dict(defined_parents=len(vals),undefined_parents=len(parents)-len(vals),positive=sum(v>EPS for v in vals),negative=sum(v< -EPS for v in vals),neutral=sum(abs(v)<=EPS for v in vals),harm_5pp=sum(v<-.05 for v in vals),harm_20pp=sum(v<-.2 for v in vals))
                        block['rank'][outcome]={f:rank_stats(parents,f,outcome) for f in SIGNALS+['teacher_quality']}
                        block['matched'][outcome]={f:matched_rank(parents,f,outcome) for f in SIGNALS if f!='holdout_agreement_gain'}
                        block['cross'][outcome]=dict(mean_quality=cross_table(parents,outcome),fully_labeled_all_anchors=cross_table(parents,outcome,True))
                    for k in ['holdout_anchors','training_anchors','teacher_labeled_anchors','common_U_frames','own_U_frames','teacher_quality','common_U_delta','own_U_delta']:
                        vals=[r[k] for r in parents if r.get(k) is not None]
                        if vals:
                            x=np.array(vals);idx=np.random.default_rng(SEED).integers(len(x),size=(BOOT,len(x)))
                            block['mean'][k]=dict(mean=float(x.mean()),ci95=np.quantile(x[idx].mean(1),[.025,.975]).tolist(),parents=len(x),undefined_parents=len(parents)-len(x))
                        else:block['mean'][k]=dict(mean=None,ci95=None,parents=0,undefined_parents=len(parents))
                    result[ds][scope][subset][order]=block
    return result

def contracts():
    from sklearn.metrics import roc_auc_score,average_precision_score
    tests=[([0,1,2,3],[0,1,0,1]),([1,1,1,1],[0,1,0,1]),([.2,.7,.7,.1],[1,0,1,0])]
    for s,y in tests:
        a,p,_=rank_distribution(s,y,np.ones((1,len(y))))
        assert abs(a[0]-roc_auc_score(y,s))<1e-14 and abs(p[0]-average_precision_score(y,s))<1e-14
    assert float(iou([0,0,1,1],[0,0,1,1]))==1
    assert float(iou([0,0,0,0],[0,0,0,0]))==0
    b=np.array([[.5,.5,.2,.2],[.5,.5,.2,.2]]);a=[dict(position=0,box=b[0],score=.9)]
    assert visible(b,b,a,a,[0],False)['holdout_agreement_gain'] is None
    assert parent_records([dict(parent_id=0,common_U_delta=1),dict(parent_id=0,common_U_delta=-1)])[0]['common_U_delta']==0
    r=rank_stats([dict(s=1,y=0),dict(s=None,y=1)],'s','y');assert r['AUROC'] is None and r['eligible_parents']==0
    return dict(status='pass',scope='CPU synthetic ROC/AP ties versus installed sklearn, degenerate IoU, empty heldout, paired parent mean and undefined binary outcome; no GPU/model/GT',contracts=8,CPU_only=True,GT_read=False)
def prepare():
    old_verify();assert read(OLD/'FINAL_COMPLETION.json')['status']=='complete'
    assert read(B/'ROOT_READBACK.json')['status']=='pass' and read(B/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed'
    assert not BASE.exists();BASE.mkdir();OUT.mkdir(parents=True,exist_ok=False)
    records=read(OLD/'DESIGN_LOCK.json')['records'];assert len(records)==512
    write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_signal_seal_and_GT',records=records,cells=512,saved_updates=1024,parents=256,signals=SIGNALS,quality_threshold=.5,neutral_tolerance=EPS,bootstrap=BOOT,seed=SEED,GT_for_feature_or_threshold_selection=False,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,production_promoted=False,history_exposed=True,time=time.time()))
    write(BASE/'CPU_CONTRACTS.json',contracts())
    own=['protocols/stvg_opd_evidence_actionability_audit_v1.md','scripts/stvg_opd_evidence_actionability_audit_v1.py','scripts/audit_stvg_opd_actionability_public_v1.py']
    original=[OLD/'DESIGN_LOCK.json',OLD/'RUNTIME.json',OLD/'FINAL_COMPLETION.json',A/'GLOBAL_READOUT_BARRIER.json',B/'EXPERT_EVIDENCE_BARRIER.json',B/'GLOBAL_PREDICTION_BARRIER.json',B/'ROOT_READBACK.json',B/'RUNTIME.json']
    write(BASE/'RUNTIME.json',dict(status='locked_cache_only_before_new_GT',pins={f:sha(ROOT/f) for f in own},original_bindings={str(p.relative_to(ROOT)):sha(p) for p in original},design_sha256=sha(BASE/'DESIGN_LOCK.json'),contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),CPU_only=True,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_actual_all1024_signal_seal',CPU_only=True,GT_read=False,time=time.time()))
def verify():
    old_verify();r=read(BASE/'RUNTIME.json')
    assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256'] and sha(BASE/'CPU_CONTRACTS.json')==r['contracts_sha256']
    for f,h in {**r['pins'],**r['original_bindings']}.items():assert sha(ROOT/f)==h,f
    return r
def extract(record):
    import torch
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    r=record;orig=ROOT/r['original_Full']['path'];ip=ROOT/r['original_Full']['input_path']
    assert sha(orig)==r['original_Full']['sha256'] and sha(ip)==r['original_Full']['input_sha256']
    old=load(orig);inp=normalized(load(ip));ep=B/'expert_I1'/f"{r['capture_index']:05}.pt"
    assert sha(ep)==read(B/'EXPERT_EVIDENCE_BARRIER.json')['files'][str(ep.relative_to(ROOT))];ex=load(ep)
    cp=B/'formal'/f"{r['public_cell']:05}"/'C_temporal_guided_OPD_repeat0.pt'
    assert sha(cp)==read(B/'GLOBAL_PREDICTION_BARRIER.json')['files'][str(cp.relative_to(ROOT))];c=load(cp)
    assert not old['fit']['GT_used'] and not c['fit']['GT_used'] and not inp['GT_read']
    assert torch.equal(old['fit']['before'],c['fit']['before'])
    assert all(torch.equal(v,c['fit']['initial'][k]) for k,v in old['fit']['initial'].items())
    assert inp['frame_ids']==ex['frame_ids']==c['frame_ids'] and inp['pixel_sha256']==ex['pixel_sha256']==c['pixel_sha256']
    assert inp['source_model_state_sha256']==c['source_model_sha256'] and old['fit']['config']==c['fit']['config']==r['config']
    before=old['fit']['before'].cpu().numpy();after=[old['fit']['final'].cpu().numpy(),c['fit']['final'].cpu().numpy()]
    experts=[inp['expert'],ex['expert']];anchors=[e['anchors']['single4'] for e in experts];visited=[e['actual_observation_positions'] for e in experts]
    assert before.shape==(len(inp['frame_ids']),4) and all(v.shape==before.shape and np.isfinite(v).all() for v in after)
    for j in [0,1]:
        assert old['fit']['positions']==[int(a['position']) for a in anchors[0]] if j==0 else c['fit']['positions']==[int(a['position']) for a in anchors[1]]
        assert all(int(a['position']) in visited[j] for a in anchors[j])
        assert all(inp['frame_ids'][int(a['position'])]==a['frame_id'] for a in anchors[j])
    signals=[visible(before,after[j],anchors[j],anchors[1-j],visited[j],r['dataset']=='hc2') for j in [0,1]]
    return dict(**{k:r[k] for k in META},before=before,after=after,anchors=anchors,visited=visited,frame_ids=inp['frame_ids'],signals=signals,
        source_model_sha256=inp['source_model_state_sha256'],pixel_sha256=inp['pixel_sha256'],source_bindings={str(p.relative_to(ROOT)):sha(p) for p in [orig,ip,ep,cp]},GT_read=False)
def seal():
    import torch
    torch.set_num_threads(2);verify();assert not torch.cuda.is_initialized();bindings={};coord=0;holdout=0
    for r in read(BASE/'DESIGN_LOCK.json')['records']:
        z=extract(r);p=BASE/'readouts'/f"{r['public_cell']:05}.pt";save(p,z);again=load(p)
        assert np.array_equal(z['before'],again['before']) and all(np.array_equal(a,b) for a,b in zip(z['after'],again['after']))
        assert z['signals']==again['signals'];coord+=z['before'].size+sum(x.size for x in z['after']);holdout+=sum(s['holdout_anchors']>0 for s in z['signals'])
        bindings[str(p.relative_to(ROOT))]=sha(p)
        status(BASE/'STATUS.json',dict(status='sealing_all_test_visible_signals',done=len(bindings),total=512,CPU_only=True,GT_read=False,time=time.time()))
    verify();assert len(bindings)==512 and not torch.cuda.is_initialized()
    write(BASE/'GLOBAL_READOUT_BARRIER.json',dict(status='sealed',cells=512,saved_updates=1024,files=bindings,coordinate_checks=coord,heldout_defined_updates=holdout,runtime_sha256=sha(BASE/'RUNTIME.json'),new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,CPU_only=True,GT_read=False,time=time.time()))
def require_seal():
    verify();b=read(BASE/'GLOBAL_READOUT_BARRIER.json');assert b['status']=='sealed' and b['cells']==512 and b['saved_updates']==1024 and not b['GT_read'];assert b['runtime_sha256']==sha(BASE/'RUNTIME.json');return b
def assemble(independent=False):
    import torch
    torch.set_num_threads(2);barrier=require_seal();assert not torch.cuda.is_initialized()
    from scripts.score_stvg_opd_temporal_capture_p7a_v1 import context_truth
    from scripts.run_decota_paper_main_v1 import read_row
    from scripts.audit_stvg_opd_actionability_public_v1 import private_labels
    gt,spans=context_truth();result=[];counts=collections.Counter()
    for r in read(BASE/'DESIGN_LOCK.json')['records']:
        p=BASE/'readouts'/f"{r['public_cell']:05}.pt";h=sha(p);assert h==barrier['files'][str(p.relative_to(ROOT))];z=load(p)
        if independent:
            fresh=extract(r);assert fresh['signals']==z['signals'] and np.array_equal(fresh['before'],z['before']) and all(np.array_equal(a,b) for a,b in zip(fresh['after'],z['after']));counts['original_source_second_hash_checks']+=len(z['source_bindings']);counts['original_coordinate_checks']+=z['before'].size+sum(a.size for a in z['after'])
        raw=read_row(r['dataset'],r['query_ordinal']);assert raw['frame_ids']==z['frame_ids'];truth=gt[r['dataset']][r['query_ordinal']];span=spans[r['dataset']][r['query_ordinal']]
        for j in [0,1]:
            row=dict(**{k:r[k] for k in META},scope='Omega'+str(j),**z['signals'][j])
            labels=private_labels(z,j,raw,truth,span,independent)
            row.update(labels);result.append(row);counts['actual_updates']+=1
        assert sha(p)==h;counts['second_readout_hash_checks']+=1
        status(BASE/('ROOT_STATUS.json' if independent else 'STATUS.json'),dict(status='independent_full_signal_label_readback' if independent else 'offline_full_GT_label_diagnostic',done=len(result),total=1024,CPU_only=True,GT_read=True,time=time.time()))
    assert len(result)==1024 and not torch.cuda.is_initialized();verify();return result,dict(counts)
def score():
    require_seal();write(BASE/'GT_EXPOSURE.json',dict(status='offline_GT_after_all1024_test_visible_signals_and_cache_readouts_sealed',GT_for_selection_or_training_or_thresholds=False,barrier_sha256=sha(BASE/'GLOBAL_READOUT_BARRIER.json'),CPU_only=True,time=time.time()))
    rr,c=assemble();put_rows(OUT/'ROWS.jsonl.gz',rr);s=aggregate(rr);write(OUT/'SUMMARY.json',s)
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_independent_root_PDF_view_publication_archive',cells=512,saved_updates=1024,parents=256,counts=c,CPU_only=True,GT_read=True,time=time.time()))
def root():
    from scripts.audit_stvg_opd_actionability_public_v1 import near,independent_summary
    rr,c=assemble(True);checks=near(rr,rows(OUT/'ROWS.jsonl.gz'));checks+=independent_summary(rr,read(OUT/'SUMMARY.json'))
    write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='all1024 cached update signals/common Before/state/source/frame/expert/after coordinates, independent GT labels and parent clustered ROC/AP/bootstrap; original complete fit mathematics reused, no model Jacobian replay',counts=c,independent_scalar_checks=checks,cells=512,saved_updates=1024,parents=256,original_complete_math_state_root_reused=True,new_model_Jacobian_replayed=False,CPU_only=True,GT_read=True,time=time.time()))
    s=read(OUT/'SUMMARY.json');eligible=True
    for ds in ['hc2','vidstg']:
        for scope in ['Omega0','Omega1']:
            for subset in ['all128','added96']:
                b=s[ds][scope][subset]['both'];v=b['rank']['common_U_delta']['holdout_agreement_gain'];ci=v['AUROC_bootstrap']
                eligible &= bool(ci and ci['ci95'] and ci['ci95'][0]>.5)
                for f in ['detector_score','before_training_agreement']:
                    m=b['matched']['common_U_delta'][f];eligible &= bool(m['ci95'] and m['ci95'][0]>0)
    write(BASE/'CANDIDATE_DECISION.json',dict(status='actual_predeclared_information_criterion_evaluated_after_full_root',heldout_signal_supports_bounded_counterfactual_selection_test=bool(eligible),new_gate_implemented=False,GT_fitted_threshold=False,dynamic_writeback_implemented=False,production_promoted=False,new_GPU_fits=0,CPU_only=True,GT_read=True,time=time.time()))
    status(BASE/'STATUS.json',dict(status='pending_actual_PDF_pixel_view_publication_archive',done=1024,total=1024,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':{'prepare':prepare,'seal':seal,'score':score,'root':root}[sys.argv[1]]()
