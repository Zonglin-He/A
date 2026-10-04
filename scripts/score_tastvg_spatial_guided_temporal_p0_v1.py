"""CPU temporal scoring only after both deployable and GT-spatial seals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,load
from scripts.spatial_guided_temporal_import_v1 import core
from scripts.audit_tastvg_dta_oracle_r1_v1 import summary
from scripts.run_tastvg_spatial_guided_temporal_p0_v1 import BASE,PUB,POOL,R2,verify,key,name,state

def tails(rows,arm):
    d=[r[arm+'_minus_Full_t'] for r in rows]
    return dict(gain=sum(x>1e-12 for x in d),harm=sum(x< -1e-12 for x in d),severe_harm_gt5pp=sum(x<-.05 for x in d),
        severe_harm_gt20pp=sum(x<-.20 for x in d),new_disjoint=sum(r['Full_t']>0 and r[arm+'_t']==0 for r in rows),
        success_destroyed=sum(r['Full_t']>.5 and r[arm+'_t']<=.5 for r in rows),
        success_rescued=sum(r['Full_t']<=.5 and r[arm+'_t']>.5 for r in rows))

def summarize(rows):
    fields=core.FIELDS+[a+'_oracle_t' for a in core.ARMS]+['GT_extrapolated_fraction','A_full_fallback_fraction']
    out={}
    for ds in ['vidstg','hc2']:
        out[ds]={}
        for sp in ['search','confirm']:
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp];out[ds][sp]={}
            for panel in ['corrupt','clean','all']:
                q=[r for r in rr if panel=='all' or (panel=='clean')==(r['condition']=='clean')]
                z=summary(q,fields);z['tails']={a:tails(q,a) for a in ['A_ROI','GT_ROI']}
                z['orders']={o:summary([r for r in q if r['order']==o],fields) for o in ['order1','order2']}
                out[ds][sp][panel]=z
    return out

def score():
    tick=time.monotonic();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='sealed' and not b['temporal_GT_scoring_started']
    verify(all_inputs=True);cells=read(BASE/'COHORT.json')['cells'];full=read(BASE/'FULL_INPUT.json')
    gtmeta=read(BASE/'GT_BOXES_INPUT.json')['metadata'];labels={}
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            f=POOL/ds/f'GT_LABELS_{sp}.json';assert sha(f)==read(BASE/'RUNTIME_LOCK.json')['GT_inputs'][str(f.relative_to(ROOT))]
            labels[(ds,sp)]=read(f)
    rows=[];predictions=[];private=[]
    for c in cells:
        k=key(c);s={};supports={};receipts={}
        fc=load(ROOT/full[k]['full_cache']);s['Full']=full[k]['Full'];supports['Full']=fc['proposals']
        for a in ['A_ROI','GT_ROI']:
            r=read(BASE/a/(name(c)+'.json'));assert sha(BASE/r['cache'])==r['cache_sha256'];v=load(BASE/r['cache'])
            assert core.top(v['proposals'],v['confidence'])==r['selection'];s[a]=r['selection'];supports[a]=v['proposals'];receipts[a]=r
        interval=labels[(c['dataset'],c['split'])][str(c['parent'])]['span']
        plan=read(ROOT/'artifacts/tastvg_current_correction_views_v1'/c['dataset']/'PLAN.json')['rows'][c['parent']]
        ids=plan['frame_ids'];origin=ids[0];length=ids[-1]-origin+1
        normalized={a:dict(v,interval=[(x-origin)/length for x in v['interval']]) for a,v in s.items()}
        pr=dict(cell_key=k,dataset=c['dataset'],split=c['split'],condition=c['condition'],order=c['order'],
            source_id=c['parent'],arrival=c['arrival'],selections=normalized,interval_coordinates='clip-normalized, entire original window',
            A_state_pre_sha256=c['pre_sha'],A_state_post_sha256=c['post_sha'],original_pixel_sha256=c['pixel_sha256'],
            Full_input_sha256=full[k]['full_input_sha256'],
            ROI_input_sha256={a:receipts[a]['feature_input_sha256'] for a in receipts},
            Full_A_ROI_GT_used=False,GT_ROI_spatial_GT_used=True,GT_ROI_temporal_span_used=False)
        r=dict(pr,**core.metrics(dict(selections=s),interval),
            **{a+'_oracle_t':max(core.iou(p,interval) for p in supports[a]) for a in core.ARMS},
            GT_extrapolated_fraction=gtmeta[k]['extrapolated_frames']/gtmeta[k]['sampled_frames'],
            A_full_fallback_fraction=receipts['A_ROI']['invalid_box_full_fallbacks']/len(ids),
            temporal_GT_scored_after_global_seal=True)
        rows.append(r);predictions.append(pr);private.append(dict(cell_key=k,selections=s))
    write(BASE/'SEALED_PHYSICAL_SELECTION_READBACK.json',private)
    # Anonymous selections are exported after execution; their private antecedents
    # and raw prediction/cache receipts were immutable before GT scoring.
    write(PUB/'PREDICTIONS.json',predictions);write(PUB/'ROWS.json',rows);out=summarize(rows);write(PUB/'SUMMARY.json',out)
    panels={ds+'/'+sp:out[ds][sp]['corrupt']['metrics']['A_ROI_minus_Full_t'] for ds in out for sp in out[ds]}
    gtpanels={ds+'/'+sp:out[ds][sp]['corrupt']['metrics']['GT_ROI_minus_Full_t'] for ds in out for sp in out[ds]}
    decision=dict(status='GO_TEACHER_EVIDENCE_ONLY' if all(x['ci95'][0]>0 for x in panels.values()) else 'NO_GO_CURRENT_CROP',
        A_ROI_panel_pass={p:x['ci95'][0]>0 for p,x in panels.items()},
        GT_ROI_panel_pass={p:x['ci95'][0]>0 for p,x in gtpanels.items()},primary=panels,GT_diagnostic=gtpanels,
        DTA_started=False,production_promoted=False,universal_impossibility_claim=False,
        GT_limit='Nearest-coordinate extension outside annotated spatial support, not perfect full-video tracking',
        interpretation='Teacher-only study; top1 interval quality, not final STVG or adaptation gain')
    write(PUB/'DECISION.json',decision)
    for f in ['DEPLOYABLE_BARRIER.json','GLOBAL_PREDICTION_BARRIER.json','SMOKE.json','SMOKE_ROOT_ACCEPTANCE.json']:
        z=read(BASE/f)
        if f=='GLOBAL_PREDICTION_BARRIER.json':z['publication_note']='Private pre-score barrier; anonymous prediction export generated afterward, never treated as an earlier seal'
        write(PUB/f,z)
    resources=dict(Full_cached_unique_inputs=len({full[key(c)]['full_cache_sha256'] for c in cells}),
        A_ROI=read(BASE/'A_ROI_BARRIER.json'),GT_ROI=read(BASE/'GT_ROI_BARRIER.json'),
        smoke=read(BASE/'SMOKE.json'),CPU_score_wall_seconds=time.monotonic()-tick,
        budget_max_new_ROI_requests=576,backbone_calls=0,spatial_expert_calls=0,backward_calls=0,parameter_updates=0)
    # Do not publish private cache paths or per-file receipt identities.
    for a in ['A_ROI','GT_ROI']:resources[a].pop('receipts')
    write(PUB/'RESOURCES.json',resources)
    fields=['cell_key','dataset','split','condition','order','source_id']+core.FIELDS+[a+'_oracle_t' for a in core.ARMS]
    with (PUB/'ROWS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    write(PUB/'SCORE_COMPLETION.json',dict(status='completed',cells=288,arm_predictions=864,
        private_global_seal_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        hashes={f:sha(PUB/f) for f in ['CONFIG.json','CODE_BINDINGS.json','PREDICTIONS.json','ROWS.json','SUMMARY.json','DECISION.json','ROWS.csv']},
        time=time.time(),CPU_wall_seconds=time.monotonic()-tick))
    state('completed_pending_root_audit_publication',cells=288,arm_predictions=864)
    print(decision['status'],{k:100*v['mean'] for k,v in panels.items()},flush=True)

if __name__=='__main__':score()
