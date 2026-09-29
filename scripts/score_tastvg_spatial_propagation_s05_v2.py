"""S0.5 post-seal whole-tube oracle, native-six-layer comparator and evidence audit."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_spatial_propagation_s05_v2 import OUT,S0,OLD,verify,CONDS
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats
from vg_tta.box_stability_diagnostics_v1 import overlap
from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes


def macro(rr,key):
    d=collections.defaultdict(list)
    for r in rr:
        v=r[key]
        if v is not None:d[r['parent']].append(v)
    return stats([np.mean(d[k]) for k in sorted(d)])


def summary(rr):
    keys=['native_s','expanded_oracle_s','six_layer_oracle_s','union_oracle_s','expanded_gain','six_layer_gain','expanded_minus_six_layer','union_minus_six_layer','step1_gain','step2_gain','step3_gain','unobserved_gain_at_s_oracle','observed_gain_at_s_oracle','expert_observed_s','native_expert_observed_s','v_gain_at_s_oracle']
    return dict(cells=len(rr),sources=len({r['parent'] for r in rr}),metrics={k:macro(rr,k) for k in keys},chosen_step=dict(collections.Counter(r['best_step'] for r in rr)),coverage=dict(nonempty_cells=sum(r['valid_evidence_frames']>0 for r in rr),valid_frames=sum(r['valid_evidence_frames'] for r in rr),sampled_frames=5*len(rr),no_seg=sum(r['mask_count']==0 for r in rr),reused=sum(r['reused'] for r in rr)),step3_harms_gt5pp=sum(r['step3_gain']<-.05 for r in rr),expanded_gain_gt1pp=sum(r['expanded_gain']>.01 for r in rr),expanded_gain_gt5pp=sum(r['expanded_gain']>.05 for r in rr))


def run():
    import ijson
    torch.set_num_threads(4);p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert len(bar['files'])==96
    for f,h in bar['files'].items():assert sha(OUT/f)==h
    eb=read(S0/'EXPERT_BARRIER.json')
    for f,h in bar['dense_files'].items():assert sha(OUT/f)==h
    for f,h in eb['files'].items():assert sha(S0/f)==h
    ra=read(OUT/'REINSERTION_AUDIT.json');assert ra['status']=='pass' and len(ra['cells'])==2 and all(c['relative_to_native']>0 and c['full_pipeline_exact'] for c in ra['cells'])
    labelpath=ROOT/'artifacts/tastvg_corruption_c0c1_v1/GT_SUBSET.json';keys={r['key'] for r in p['rows']};gt={}
    with labelpath.open('rb') as f:
        for k,v in ijson.kvitems(f,'',use_float=True):
            if k in keys:gt[k]=v
    assert set(gt)==keys
    write(OUT/'GT_EXPOSURE.json',dict(retained_queries=16,historically_exposed=True,GT_for_scoring_only=True,container_sha256=sha(labelpath),prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),time=time.time()))
    rows=[];calls=0;reinsertion=0;loss_checks=0
    for row in p['rows']:
        label=gt[row['key']];valid=np.asarray(label['valid'],bool)
        for cond in CONDS:
            name=f"{row['ordinal']:03}.pt";x=load(OUT/'ta'/cond/name);e=load(S0/'expert'/cond/name);dense=load(OUT/'dense'/cond/name);old=load(OLD/'capture'/cond/name)
            assert x['pixel_sha256']==e['pixel_sha256']==old['pixel_sha']
            vb,bb=mask_boxes(e['masks'],e['positions'],len(row['frame_ids']));assert np.array_equal(vb,e['valid']) and np.array_equal(bb,e['boxes'])
            native=old['native'];base=x['trajectory'][0]['prediction'];assert torch.equal(base['boxes'],native['boxes'])
            assert all(torch.equal(a,b) for a,b in zip(base['logits'],native['logits']))
            metrics=[];quality=[]
            for step in x['trajectory']:
                pred=step['prediction'];m,q=checked_score(pred['boxes'],label,row['frame_ids'],pred['indices']);calls+=1;metrics.append({k:m[k] for k in ['sIoU','tIoU','vIoU_corrected']});quality.append(q)
            sm=[]
            for candidate in old['candidates']['spatial']:
                m,_=checked_score(candidate['boxes'],label,row['frame_ids'],native['indices']);calls+=1;sm.append(m['sIoU'])
            assert any(torch.equal(c['boxes'],base['boxes']) for c in old['candidates']['spatial'])
            best=int(np.argmax([m['sIoU'] for m in metrics]));b=metrics[0]['sIoU'];so=max(sm);exp=metrics[best]['sIoU'];assert exp>=b and so>=b
            seen=np.zeros(len(valid),bool);seen[e['positions']]=True;obs=seen&valid;unobs=~seen&valid;ev=vb&valid
            qexpert=overlap(e['boxes'],label['boxes']);qgain=quality[best]-quality[0]
            for j,d in enumerate(x['diagnostics']):
                assert d['appearance_only_exact'] and d['relative_to_native']<=.004*(j+1)+1e-7
                assert d['relative_step']==0 or abs(d['relative_step']-.004)<1e-7
                for idx,key in [(j,'loss_before'),(j+1,'loss_after')]:
                    # Independent scalar no-clamp GIoU, not the adaptation function.
                    vals=[]
                    pp=x['trajectory'][idx]['prediction']['boxes'].double().numpy()
                    for i in np.flatnonzero(dense['valid']):
                        a=pp[i];bq=np.asarray(dense['boxes'][i],float);lo=a[:2]-a[2:]/2;hi=a[:2]+a[2:]/2;ql=bq[:2]-bq[2:]/2;qh=bq[:2]+bq[2:]/2
                        inter=np.maximum(np.minimum(hi,qh)-np.maximum(lo,ql),0).prod();u=a[2:].prod()+bq[2:].prod()-inter;c=(np.maximum(hi,qh)-np.minimum(lo,ql)).prod()
                        vals.append(x['coefficients'][0]*abs(a-bq).sum()+x['coefficients'][1]*(1-inter/u+(c-u)/c))
                    val=float(np.sum(vals))/len(pp) if vals else 0.;assert abs(val-d[key])<3e-5,(row['ordinal'],cond,j,key,val,d[key]);loss_checks+=1
            if 'reinsertion' in x:assert x['reinsertion']['full_pipeline_exact'];reinsertion+=1
            rows.append(dict(propagation=dense['diagnostics'],dense_valid_frames=int(dense['valid'].sum()),unobserved_step_gains=[float((q-quality[0])[unobs].mean()) if unobs.any() else None for q in quality],observed_step_gains=[float((q-quality[0])[obs].mean()) if obs.any() else None for q in quality],dense_unobserved_s=float(overlap(dense['boxes'],label['boxes'])[unobs&dense['valid']].mean()) if (unobs&dense['valid']).any() else None,native_dense_unobserved_s=float(quality[0][unobs&dense['valid']].mean()) if (unobs&dense['valid']).any() else None,unobserved_target_valid_frames=int((unobs&dense['valid']).sum()),parent=row['ordinal'],condition=cond,native_s=b,expanded_oracle_s=exp,six_layer_oracle_s=so,union_oracle_s=max(so,exp),expanded_gain=exp-b,six_layer_gain=so-b,expanded_minus_six_layer=exp-so,union_minus_six_layer=max(so,exp)-so,**{f'step{j}_gain':metrics[j]['sIoU']-b for j in range(1,4)},best_step=best,step_metrics=metrics,six_layer_s=sm,six_layer_count=len(sm),valid_evidence_frames=int(vb.sum()),mask_count=e['mask_count'],reused=e['reused_from'] is not None,expert_observed_valid_frames=int(ev.sum()),expert_observed_s=float(qexpert[ev].mean()) if ev.any() else None,native_expert_observed_s=float(quality[0][ev].mean()) if ev.any() else None,observed_gain_at_s_oracle=float(qgain[obs].mean()) if obs.any() else None,unobserved_gain_at_s_oracle=float(qgain[unobs].mean()) if unobs.any() else None,v_gain_at_s_oracle=(metrics[best]['vIoU_corrected']-metrics[0]['vIoU_corrected']) if metrics[best]['vIoU_corrected'] is not None and metrics[0]['vIoU_corrected'] is not None else None,changed_interval_by_step=[z['prediction']['indices']!=native['indices'] for z in x['trajectory']],diagnostics=x['diagnostics']))
    summaries={c:summary([r for r in rows if (r['condition']!='clean' if c=='corruption' else r['condition']==c)]) for c in ['corruption']+CONDS}
    write(OUT/'ROWS.json',rows);write(OUT/'SUMMARY.json',summaries)
    write(OUT/'AUDIT.json',dict(status='pass',cells=96,dual_metric_calls=calls,independent_loss_checks=loss_checks,mask_to_box_checks=96,full_reinsertion_checks=reinsertion,edited_reinsertion_checks=len(ra['cells']),all_native_retained=True,all_Ha_only=True,GT_post_seal=True,time=time.time()))
    print({c:summaries[c]['metrics']['expanded_gain'] for c in ['corruption','clean']})

if __name__=='__main__':run()
