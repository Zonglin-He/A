"""Publish completed fixed five-point LR selection; saved scalar evidence only."""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import export_stvg_lite_batch001 as e
from scripts import publish_stvg_soft_ep_v1 as publisher
BASE=e.BASE
OUT=ROOT/'results/stvg_lite_event_support/2026-10-16/batch002'
PUB=BASE/'publication/batch002'
DOC=ROOT/'docs/STVG_LITE_EVENT_SUPPORT_SEARCH002.md'
FACTORS={'0.8':'LR_080','0.9':'LR_090','1.0':'SCOPE_FG','1.1':'LR_110','1.2':'LR_120'}


def export():
    assert not (OUT/'BATCH_RECEIPT.json').exists()
    lock=e.read(BASE/'FINAL_RECIPE_LOCK.json')
    assert lock['status']=='locked' and not lock['suite_final'] and not lock['CURRENT_promoted']
    assert lock['selected_spatial_recipe']=='lite_fullstep' and lock['selected_temporal']=='temporal_lite'
    assert lock['selected_response_scope']=='full_grid' and not lock['GT_used_for_online_reference_or_step']
    assert lock['structural_selection_lock_sha256']==e.sha(BASE/'STRUCTURAL_SELECTION_LOCK.json')
    assert lock['selected_temporal_lock_sha256']==e.sha(BASE/'SELECTED_TEMPORAL_LOCK.json')
    raw={};designs={};means={ds:{} for ds in ['hc2','vidstg']};roots={}
    for f,tid in FACTORS.items():
        raw[tid],roots[tid]=e.load_block(BASE/'trials'/tid)
        d=e.read(BASE/'trials'/tid/'DESIGN_LOCK.json');designs[tid]=d
        assert d['recipe']==lock['selected_spatial_recipe'] and d['response_scope']=='full_grid'
        assert d['selected_temporal_lock_sha256']==lock['selected_temporal_lock_sha256']
        assert d['structural_selection_sha256']==lock['structural_selection_lock_sha256']
        assert float(d['lr_multiplier'])==float(f)
        for ds in means:
            m=100*math.fsum(r['metrics']['m_vIoU'] for r in raw[tid] if r['dataset']==ds and r['arm']=='TS_M2')/64
            assert abs(m-lock['lr_search']['factor_scores_pp'][ds][f])<1e-12
            means[ds][f]=m
    for f,tid in FACTORS.items():
        for ds in means:
            cfg=dict(designs[tid]['configs'][ds]['M2']);center=dict(designs['SCOPE_FG']['configs'][ds]['M2'])
            assert math.isclose(cfg.pop('lr'),center.pop('lr')*float(f),rel_tol=1e-14) and cfg==center
            assert designs[tid]['response_settings']==designs['SCOPE_FG']['response_settings']
            assert e.sha(BASE/'trials'/tid/'REFERENCE_LOCK.json')==e.sha(BASE/'trials/SCOPE_FG/REFERENCE_LOCK.json')
            assert e.sha(BASE/'trials'/tid/'TEMPORAL_LOCK.json')==e.sha(BASE/'trials/SCOPE_FG/TEMPORAL_LOCK.json')
    for ds,values in means.items():
        maximum=max(values.values());tied=[float(f) for f,v in values.items() if abs(v-maximum)<=1e-12]
        selected=min(tied,key=lambda f:(abs(f-1),f));assert selected==lock['lr_factors_by_domain'][ds]
        cfg=designs[FACTORS[str(selected)]]['configs'][ds]['M2'];assert cfg==lock['configs'][ds]
    aliases={}
    for ds,prefix in [('hc2','HC'),('vidstg','VID')]:
        ids=sorted({r['case_id'] for r in raw['SCOPE_FG'] if r['dataset']==ds});assert len(ids)==64
        aliases.update({cid:f'{prefix}{i:03}' for i,cid in enumerate(ids,1)})
    rows=e.sanitized_rows(raw['LR_120'],aliases);center=e.sanitized_rows(raw['SCOPE_FG'],aliases)
    cost=e.read(BASE/'trials/LR_120/MATCHED_EFFICIENCY_READBACK.json')
    assert cost['status']=='complete' and cost['count']==128
    cr=[]
    for r in cost['records']:
        fit=r['fit_cost'];cr.append(dict(case_id=aliases[r['case_id']],no_fit=fit is None,
            decoder_calls=sum(fit['decode_counts'].values()) if fit else 0,
            fit_cuda_synchronized_seconds=fit['fit_wall_cuda_synchronized_seconds'] if fit else None))
    e.write(OUT/'MATCHED_COSTS.json',dict(status='complete',records=cr,
        scope='LR120 cached spatial fit and decoder only; not end-to-end deployment cost'))
    e.write(OUT/'trials/LR_120/ROWS.json',dict(status='complete',count=128,row_count=512,metrics_unit='fraction',rows=rows))
    e.write(OUT/'trials/LR_120/SUMMARY.json',dict(status='complete',summary=e.summaries(rows)))
    pairs=[]
    for a,b in [('TS_M2','F'),('TS_M2','T'),('S_M2','F')]:pairs+=e.contrast(rows,a,rows,b,a+' minus '+b)
    pairs+=e.contrast(rows,'TS_M2',center,'TS_M2','LR_120 minus SCOPE_FG')
    e.write(OUT/'trials/LR_120/PAIRED.json',dict(status='complete',paired=pairs,bootstrap_executed=False))
    d=designs['LR_120'];e.write(OUT/'trials/LR_120/CONFIGURATION.json',{k:d[k] for k in
        ['status','trial_id','recipe','response_scope','lr_multiplier','configs','response_settings']})
    e.write(OUT/'FINAL_DEVELOPMENT_RECIPE.json',{k:lock[k] for k in
        ['status','completion_scope','method_validation_status','selected_spatial_recipe','selected_response_scope',
         'configs','response_settings','selected_temporal','lr_factors_by_domain','lr_search','GT_read',
         'GT_used_for_online_reference_or_step','CURRENT_promoted','suite_final','confirmation_status','bootstrap_executed']})
    e.write(OUT/'FIVE_POINT_SELECTION.json',dict(status='complete',means_percent=means,
        selected_factors=lock['lr_factors_by_domain'],center_exact_reuse=True,center_trial='SCOPE_FG',
        independent_GPU_runs=4,cohort='128 exposed development sources, 64 per domain',
        rule='maximum per-domain mean, ties within 1e-12 pp closest to 1.0 then lower factor',
        previous_results='../batch001/',not_independent_confirmation=True))
    e.write(OUT/'ROOT_SOURCE_AND_SELECTION_READBACK.json',dict(status='pass',count=128,completed_trials=list(FACTORS.values()),
        all_pre_GT_roots_and_prediction_seals_verified=True,all_score_hash_chains_verified=True,
        only_LR_changed=True,selection_independently_recomputed=True,full_backward_recomputed=False,
        new_inference=False,new_GT_read=False,source_bindings=e.SOURCES))
    lines=['# Completed event-support and five-point learning-rate study — batch 002','','The locked search is complete. Actual re-clocked development pipelines and 64-source confirmation are now the next stage; their outcomes are not reported here. This is not whole-study FINAL or a CURRENT promotion.','',
        '## Fixed development selection','','TA-STVG cross-domain Source checkpoints, 128 exposed sources (HC64/Vid64), one query per parent. Temporal-Lite, Spatial Lite-fullstep and full-grid response support are fixed. Only spatial Adam learning rate changes; original references, clocks, stochastic keys, other parameters and acceptance rules remain matched.','',
        '| Multiplier | HC LR | HC m_vIoU (%) | Vid LR | Vid m_vIoU (%) |','|---|---:|---:|---:|---:|']
    for f in FACTORS:lines.append(f'| {f} | {designs[FACTORS[f]]["configs"]["hc2"]["M2"]["lr"]:.5f} | {means["hc2"][f]:.9f} | {designs[FACTORS[f]]["configs"]["vidstg"]["M2"]["lr"]:.5f} | {means["vidstg"][f]:.9f} |')
    lines+=['','Both domains select 0.9×: HC lr=0.0045, VidSTG lr=0.027. Relative to center, HC improves only 0.000101 pp and VidSTG 0.074072 pp; the HC difference is negligible and neither proves confirmation/generalization. The center reuses SCOPE_FG under strict identity, so five logical points are four new GPU runs.','',
        'Event-output support was not selected: it failed the prelocked equal-domain non-degradation condition. This batch does not add another parameter search. Spatial retains 1792 episodic Query/LN parameters; inherited writeback and amplitude fields do not activate LN carry or removed Lite checks. Temporal remains the original 512-parameter L-BFGS interface.','',
        'GT is used only for post-seal offline development scoring/selection, never for candidates, references or online step acceptance. All positive, negative, neutral and unknown cases are preserved. No bootstrap, official full-set run or new teacher.','',
        '[Complete first-batch results and pinned source snapshot](../results/stvg_lite_event_support/2026-10-16/batch001/)','[New LR_120 results, final development configuration and selection evidence](../results/stvg_lite_event_support/2026-10-16/batch002/)','','![Five-point learning rates](../results/stvg_lite_event_support/2026-10-16/batch002/FIVE_POINT_LR.png)','']
    DOC.write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.5),layout='constrained')
    for ax,ds in zip(axs,means):
        ax.plot([float(x) for x in FACTORS],[means[ds][x] for x in FACTORS],marker='o')
        ax.set_title(ds+' exposed development');ax.set_xlabel('Spatial learning-rate multiplier');ax.set_ylabel('m_vIoU (%)');ax.grid(alpha=.25)
    fig.savefig(OUT/'FIVE_POINT_LR.png',dpi=160);plt.close(fig)
    e.write(OUT/'BATCH_RECEIPT.json',dict(status='complete',scope='fixed five-point development search complete; not full study',
        count=128,logical_new_rows=512,all_signs_unknown_preserved=True,new_model_inference=False,new_GT_read=False,
        source_snapshot_reused_from_batch001=True,whole_study_FINAL_complete=False))


def audit():
    original=e.read(BASE/'trials/LR_120/POSTSEAL_ROWS.json')['rows'];public=e.read(OUT/'trials/LR_120/ROWS.json')['rows']
    assert len(original)==len(public)==512
    roster=e.read(BASE/'trials/SCOPE_FG/POSTSEAL_ROWS.json')['rows'];aliases={}
    for ds,prefix in [('hc2','HC'),('vidstg','VID')]:
        aliases.update({c:f'{prefix}{i:03}' for i,c in enumerate(sorted({r['case_id'] for r in roster if r['dataset']==ds}),1)})
    for a,b in zip(original,public):
        assert a['metrics']==b['metrics'] and a['dataset']==b['dataset'] and a['arm']==b['arm']
        assert aliases[a['case_id']]==b['case_id'] and a['status']==b['status']
    center=e.sanitized_rows(roster,aliases)
    for cell in e.read(OUT/'trials/LR_120/PAIRED.json')['paired']:
        if cell['comparison']=='LR_120 minus SCOPE_FG':a=public;aa='TS_M2';b=center;bb='TS_M2'
        else:a=b=public;aa,bb=cell['comparison'].split(' minus ')
        ds=cell['dataset'];m=cell['metric']
        x={r['case_id']:r['metrics'][m] for r in a if r['dataset']==ds and r['arm']==aa}
        y={r['case_id']:r['metrics'][m] for r in b if r['dataset']==ds and r['arm']==bb}
        delta={c:(x[c]-y[c])*100 for c in sorted(x) if x[c] is not None and y[c] is not None}
        assert delta==cell['per_source_delta_pp'] and len(delta)==cell['n_pairs']
        assert sorted(set(x)-set(delta))==cell['unknown_case_ids']
        assert [c for c,v in delta.items() if v>1e-10]==cell['positive_case_ids']
        assert [c for c,v in delta.items() if v<-1e-10]==cell['negative_case_ids']
        assert [c for c,v in delta.items() if abs(v)<=1e-10]==cell['neutral_case_ids']
    final=e.read(OUT/'FINAL_DEVELOPMENT_RECIPE.json');assert final['suite_final'] is False and final['CURRENT_promoted'] is False
    forbidden={'query_text','caption','raw_output','video_path','physical_frame_ids','box_xyxy','box_cxcywh','weights','optimizer_state','GT_coordinates'}
    def scan(v):
        if isinstance(v,dict):
            for k,x in v.items():assert k not in forbidden,k;scan(x)
        elif isinstance(v,list):
            for x in v:scan(x)
        elif isinstance(v,str):assert '/home/' not in v and '.codex/attachments' not in v
    for p in OUT.rglob('*.json'):scan(e.read(p))
    for rel,h in e.read(OUT/'ROOT_SOURCE_AND_SELECTION_READBACK.json')['source_bindings'].items():assert e.sha(ROOT/rel)==h
    proof=dict(status='pass',rows_checked=512,metric_values_checked=2560,all_values_exact=True,GT_read=False,CPU_only=True)
    e.write(OUT/'ROOT_PUBLIC_SCALAR_AUDIT.json',proof);print(json.dumps(proof))


def paths():
    assert json.loads((OUT/'ROOT_PUBLIC_SCALAR_AUDIT.json').read_text())['status']=='pass'
    return {str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}|{str(DOC.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['export','audit','stage','verify']);a=parser.parse_args()
    PUB.mkdir(parents=True,exist_ok=True)
    if a.mode=='export':export();audit()
    elif a.mode=='audit':audit()
    else:
        publisher.BASE=PUB;publisher.expected_paths=paths
        if a.mode=='stage':publisher.stage(publisher.DEFAULT_CHECKOUT,None,False)
        else:publisher.verify_stage(publisher.DEFAULT_CHECKOUT)

if __name__=='__main__':main()
