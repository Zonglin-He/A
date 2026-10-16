"""Post-hoc GT diagnosis of sealed Paper48 P1/P5; CPU geometry only, no model."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
import sys, json, hashlib, time, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from vg_tta.tastvg_paper48_metrics_v1 import xyxy, source_summary

BASE = ROOT/'artifacts/tastvg_pipeline_diagnosis_cpu_v1'
EPS = 1e-10
def read(p): return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), p
    p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def status(x): (BASE/'STATUS.json').write_text(json.dumps(x, indent=2)+'\n')

def truth(panel, p):
    parents = sorted(set(next(iter(p['orders'].values()))))
    dense, spans = {}, {}
    if panel == 'P1':
        import ijson
        from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations, trajectory_for_target
        lp=ROOT/'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json'
        ap=ROOT/'external/VidSTG-Dataset/annotations/test_annotations.json'
        vp=ROOT/'downloads/vidor/validation-annotation.zip'
        keys={p['rows'][i]['key'] for i in parents}
        with lp.open('rb') as f: gt={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
        annotations=read(ap); raw=load_vidor_annotations(vp,None)
        for i in parents:
            row=p['rows'][i]; a=gt[row['key']]['official_annotation']; v=annotations[a['annotation_index']]
            assert v['vid']==row['source'] and v[a['field']][a['query_index']]['target_id']==a['target_id']
            assert v[a['field']][a['query_index']]['description']==row['input']['caption']
            spans[i]=[v['temporal_gt']['begin_fid'],v['temporal_gt']['end_fid']]
            dense[i]={}
            for fid,rec in trajectory_for_target(raw[row['source']],a['target_id']).items():
                if spans[i][0]<=int(fid)<spans[i][1]:
                    x,y,w,h=rec['bbox'];dense[i][int(fid)]=[x,y,x+w,y+h]
        provenance={str(q.relative_to(ROOT)):sha(q) for q in [lp,ap,vp]}
    else:
        lock=read(ROOT/'artifacts/tastvg_paper48_v1/P5/LOCK.json'); ap=Path(lock['metadata_intake_path'])
        assert sha(ap)==lock['metadata_intake_sha256']; annotations=read(ap)
        for i in parents:
            row=p['rows'][i]; v=annotations[row['annotation_index']]
            assert v['original_video_id']==row['input']['original_video_id'] and v['caption'].lower()==row['input']['caption']
            start=int(v['tube_start_frame'])-1; end=start+len(v['trajectory'])-1;spans[i]=[start,end]
            dense[i]={start+j:[x,y,min(x+w,row['input']['width']),min(y+h,row['input']['height'])] for j,(x,y,w,h) in enumerate(v['trajectory'])}
        provenance={str(ap.relative_to(ROOT)):sha(ap)}
    assert all(dense.values())
    return dense, spans, provenance

def evaluator(boxes, row, gt, span, clip):
    """Vectorized equivalent of audited dense_official_metrics; no extrapolation."""
    ids=np.asarray(row['frame_ids']); b=xyxy(boxes,row['input']['width'],row['input']['height'])
    if clip: b=np.maximum(b,0)
    fids=np.asarray(sorted(gt)); t=np.asarray([gt[int(f)] for f in fids])
    pred=np.column_stack([np.interp(fids,ids,b[:,k]) for k in range(4)])
    inter=np.maximum(np.minimum(pred[:,2:],t[:,2:])-np.maximum(pred[:,:2],t[:,:2]),0).prod(1)
    union=np.maximum(pred[:,2:]-pred[:,:2],0).prod(1)+np.maximum(t[:,2:]-t[:,:2],0).prod(1)-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    iou[(fids<ids[0])|(fids>ids[-1])]=0
    g,h=span
    def score(idx):
        a,z=int(ids[idx[0]]),int(ids[idx[1]])+1
        overlap=max(0,min(z,h)-max(a,g))
        return dict(t=overlap/(z-a+h-g-overlap),v=float(iou[(fids>=max(a,g))&(fids<min(z,h))].sum()/max(max(z,h)-min(a,g),1)))
    return score

def transitions(rows,a,b):
    v=np.array([r[b]-r[a] for r in rows]); out=dict(cells=len(rows),degraded=int((v< -EPS).sum()),improved=int((v>EPS).sum()),unchanged=int((abs(v)<=EPS).sum()),gross_loss_pp=float(-np.minimum(v,0).mean()*100),gross_gain_pp=float(np.maximum(v,0).mean()*100))
    for threshold in [.3,.5]:
        before=np.array([r[a]>threshold for r in rows]);after=np.array([r[b]>threshold for r in rows]);n=int(before.sum());bad=int((before&~after).sum())
        out[str(threshold)]=dict(correct_before=n,correct_to_wrong=bad,conditional_damage_rate=bad/n if n else None,wrong_before=int((~before).sum()),wrong_to_correct=int((~before&after).sum()))
    return out

def summarize(rows):
    fields=['delta_inherited_boxes','delta_inherited_interval','delta_inherited_total','delta_temporal_rerank','delta_total','delta_temporal_oracle_v','delta_temporal_oracle_t']
    out={}
    for group in ['corruption','clean']:
        rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')];out[group]={}
        for subset in ['all','expert','nonexpert']:
            seq=[r for r in rr if subset=='all' or r['expert_scheduled']==(subset=='expert')]
            z=source_summary(seq,fields)
            z['transitions']={name:transitions(seq,a,b) for name,a,b in [('inherited_boxes','frozen_v','boxes_only_v'),('inherited_interval','boxes_only_v','slow_v'),('inherited_total','frozen_v','slow_v'),('temporal_rerank','slow_v','final_v'),('total','frozen_v','final_v')]}
            if subset=='expert':
                z['spatial_updates']=dict(scheduled=len(seq),updated=sum(r['updated'] for r in seq),no_valid_expert_frames=sum(r['spatial_reason']=='no_valid_expert_frames' for r in seq),zero_gradient=sum(r['spatial_reason']=='zero_gradient' for r in seq),flat_rewards=sum(r['flat_rewards'] for r in seq),loss_decreased=sum(r['loss_decreased'] for r in seq))
                z['temporal_selection']={}
                for m in ['v','t']:
                    a=np.array([r['candidate_best_'+m] for r in seq]);b=np.array([r['final_'+m] for r in seq]);native=np.array([r['slow_'+m] for r in seq]);regret=a-b
                    z['temporal_selection'][m]=dict(suboptimal_cells=int((regret>EPS).sum()),support_no_better_than_native=int((a<=native+EPS).sum()),selected_harms_native=int((b<native-EPS).sum()),oracle_gap_pp=float(regret.mean()*100),thresholds={str(t):dict(support_has_correct=int((a>t).sum()),missed_correct=int(((a>t)&(b<=t)).sum()),miss_rate=float(((a>t)&(b<=t)).sum()/(a>t).sum()) if (a>t).any() else None,no_correct_candidate=int((a<=t).sum())) for t in [.3,.5]})
            out[group][subset]=z
    return out

def run(panel):
    source=ROOT/'artifacts/tastvg_paper48_v1'/panel;out=BASE/panel
    if (out/'COMPLETION.json').exists(): return
    p=read(source/'PLAN.json');bar=read(source/'PREDICTION_BARRIER.json');assert len(bar['files'])==bar['cells']==p['total']
    assert read(source/'AUDIT.json')['status']=='pass'
    # Verify ALL receipt and raw payload hashes before accessing diagnostic labels.
    for relative,h in bar['files'].items():
        f=source/relative;assert sha(f)==h;assert sha(f.with_suffix('.pt'))==read(f)['sha256']
    write(out/'INPUT_LOCK.json',dict(panel=panel,cells=p['total'],files={k:sha(source/k) for k in ['PLAN.json','PREDICTION_BARRIER.json','AUDIT.json','ROWS.json']},script_sha256=sha(Path(__file__)),all_payloads_verified=True,time=time.time()))
    dense,spans,provenance=truth(panel,p)
    write(out/'GT_EXPOSURE.json',dict(purpose='User-authorized posthoc pipeline diagnosis; no search feedback or model changes',sources=len(dense),GT_sources=provenance,after_original_global_panel_seal=True,time=time.time()))
    original={(r['condition'],r['order'],r['arrival']):r for r in read(source/'ROWS.json')};rows=[];maxerr=0.;checks=0
    for cond in p['conditions']:
        for order,seq in p['orders'].items():
            for arrival,parent in enumerate(seq):
                x=torch.load(source/'online'/cond/order/f'{arrival:05}.pt',map_location='cpu',weights_only=False)
                assert x['parent']==parent and x['condition']==cond and x['order']==order
                row=p['rows'][parent];fs=evaluator(x['source_native']['boxes'],row,dense[parent],spans[parent],panel=='P5');ss=evaluator(x['slow']['boxes'],row,dense[parent],spans[parent],panel=='P5')
                f=fs(x['source_native']['indices']);b=ss(x['source_native']['indices']);s=ss(x['slow']['indices']);z=ss(x['final_indices'])
                old=original[cond,order,arrival]
                for arm,m in [('Frozen',f),('Ours',z)]:
                    for short,long in [('v','m_vIoU'),('t','m_tIoU')]:
                        err=abs(m[short]-old[arm+'_'+long]);assert err<1e-10,(panel,arrival,err);maxerr=max(maxerr,err);checks+=1
                r=dict(parent=parent,order=order,condition=cond,arrival=arrival,expert_scheduled=x['expert_scheduled'],updated=x['updated'])
                for name,m in [('frozen',f),('boxes_only',b),('slow',s),('final',z)]:r.update({name+'_'+k:v for k,v in m.items()})
                r.update(delta_inherited_boxes=b['v']-f['v'],delta_inherited_interval=s['v']-b['v'],delta_inherited_total=s['v']-f['v'],delta_temporal_rerank=z['v']-s['v'],delta_total=z['v']-f['v'],delta_temporal_oracle_v=0.,delta_temporal_oracle_t=0.)
                assert abs(r['delta_total']-r['delta_inherited_total']-r['delta_temporal_rerank'])<1e-12
                if x['expert_scheduled']:
                    td=x['temporal'];assert td['candidates'][0]['indices']==x['slow']['indices'];assert x['final_indices']==td['candidates'][td['selected']]['indices'];assert int(np.argmax(td['scores']))==td['selected']
                    cm=[ss(c['indices']) for c in td['candidates']]
                    r.update(candidate_count=len(cm),selected=td['selected'],critic_scores=td['scores'],candidate_metrics=cm,candidate_best_v=max(c['v'] for c in cm),candidate_best_t=max(c['t'] for c in cm))
                    r['delta_temporal_oracle_v']=r['candidate_best_v']-z['v'];r['delta_temporal_oracle_t']=r['candidate_best_t']-z['t']
                    assert r['delta_temporal_oracle_v']>=-EPS
                    u=x.get('update');rewards=x.get('rewards');r.update(spatial_reason='updated' if x['updated'] else ('no_valid_expert_frames' if u is None else 'zero_gradient'),flat_rewards=bool(rewards is not None and np.ptp(rewards)<=1e-12),loss_decreased=bool(u is not None and u['loss_after']<u['loss_before']-EPS))
                rows.append(r)
                if len(rows)%500==0:status(dict(status='running_cpu',panel=panel,cells=len(rows),total=p['total']));print(panel,len(rows),flush=True)
    assert len(rows)==p['total'];assert not torch.cuda.is_initialized()
    write(out/'ROWS.json',rows);summary=summarize(rows);write(out/'SUMMARY.json',summary)
    cases={stage:sorted([r for r in rows if r['condition']!='clean'],key=lambda r:r[stage])[:12] for stage in ['delta_inherited_total','delta_temporal_rerank','delta_total']}
    write(out/'WORST_CASES.json',cases)
    write(out/'COMPLETION.json',dict(status='completed_cpu',cells=len(rows),endpoint_metric_checks=checks,max_endpoint_discrepancy=maxerr,GPU_initialized=False,new_model_forwards=0,source_predictions_modified=False,rows_sha256=sha(out/'ROWS.json'),summary_sha256=sha(out/'SUMMARY.json'),time=time.time()))
    print('COMPLETE',panel,len(rows),'maxerr',maxerr,flush=True)

if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    for panel in sys.argv[1:] or ['P1','P5']:run(panel)
    status(dict(status='completed_pending_root_review',panels=sys.argv[1:] or ['P1','P5'],time=time.time()))
