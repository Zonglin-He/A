"""Anonymous complete finite CRC readout, without private media or prompts."""
import collections,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_crc_common_v1 import *
OUT=ROOT/'results/stvg_contrastive_correction/2026-10-15'
def main():
    assert read(BASE/'ROOT_STUDENT_READBACK.json')['status']=='pass'
    assert read(BASE/'ROOT_POSTSEAL_SCALAR_READBACK.json')['status']=='pass'
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['POSTSEAL_ROWS.json','POSTSEAL_SUMMARY.json','PARENT_PAIRED_DELTAS.json','POSTSEAL_REFERENCE_DIAGNOSTICS.json','ROOT_STUDENT_READBACK.json','ROOT_POSTSEAL_SCALAR_READBACK.json','ROOT_TEACHER_OUTPUT_READBACK.json','TECHNICAL_DECISION.json','ROOT_NO_UPDATE_READBACK.json']:
        d=read(BASE/name)
        if name=='POSTSEAL_ROWS.json':
            for row in d['rows']: row['support']={k:v for k,v in row['support'].items() if not isinstance(v,list)}
        if name=='POSTSEAL_REFERENCE_DIAGNOSTICS.json':
            for row in d['rows']:row.pop('center_position',None)
        write(OUT/name,d)
    propagation=read(BASE/'propagation/ROOT_PROPAGATION_READBACK002.json')
    pr=[]
    for case in propagation['cases']:
        branches={}
        for name,b in case['branches'].items():
            v={k:b[k] for k in ['status','actual_fit','algorithm','accepted_optimizer_steps','proposed_optimizer_steps','rejected_optimizer_steps','anchor_count','optimizer_steps'] if k in b}
            if b.get('rounds'):
                final=b['rounds'][-1]
                for k in ['full_grid_box_delta_from_source','full_grid_box_delta_from_effective_supervision','anchor_teacher_iou_after','anchor_teacher_iou_before']:
                    if k in final:v[k]=final[k]
            branches[name]=v
        pr.append(dict(case_alias=case['case_alias'],domain=case['domain'],frame_grid_count=case['frame_grid_count'],branches=branches))
    write(OUT/'SAVED_PROPAGATION_AUDIT.json',dict(status='pass',GT_read=False,CPU_only=True,summary={k:v for k,v in propagation['summary'].items() if k!='critical_case_profiles'},interpretation=propagation['interpretation'],cases=pr,full_private_receipt_sha256=sha(BASE/'propagation/ROOT_PROPAGATION_READBACK002.json')))
    summary=read(BASE/'POSTSEAL_SUMMARY.json')['summary']
    lookup={(r['dataset'],r['arm'],r['metric']):r['mean'] for r in summary}
    write(OUT/'CONFIGURATION.json',dict(status='locked_actual',spatial_configs=read(BASE/'DESIGN_LOCK.json')['spatial_configs'],generation=read(BASE/'TEACHER_RUNTIME.json')['generation'],temporal='exact existing Offset-Trust predictions',student='TA-STVG Source, episodic reset, Query/LN 1792',teacher='STVG-R1-7B frozen unquantized BF16',candidate_pool='exact original new_crop, 1 Source plus 8 Weight-free alternatives',pairwise_outputs=['Source','Alternative','Uncertain'],bootstrap_executed=False,GT_before_prediction_seal=False))
    rows=[]
    for arm in ['F','T','S_historical','TS_historical','S_Pair','TS_Pair']:
        rows.append(dict(arm=arm,HC_vIoU_percent=100*lookup['hc2',arm,'m_vIoU'],Vid_vIoU_percent=100*lookup['vidstg',arm,'m_vIoU']))
    write(OUT/'MAIN_TABLE.json',dict(status='complete',rows=rows,development_sources_per_domain=16,independent_confirmation=False,GT_read=True))
    diag=read(BASE/'POSTSEAL_REFERENCE_DIAGNOSTICS.json')['summary']
    bar=read(BASE/'TEACHER_GLOBAL_BARRIER.json')
    decisions=collections.defaultdict(collections.Counter);cost=collections.Counter()
    for item in bar['files']:
        r=read(ROOT/item['path']);p=r['parsed']
        decisions[r['dataset']][p.get('decision',p.get('label')) if p['status']=='valid' else 'invalid']+=1
        for key in ['generate_seconds','request_seconds','input_tokens','visual_tokens','output_tokens']:
            value=r.get({'generate_seconds':'model_generate_seconds','request_seconds':'end_to_end_request_seconds'}.get(key,key),r.get('cost',{}).get(key))
            if isinstance(value,(int,float)):cost[key]+=value
    teacher=dict(status='complete',pairwise_calls=len(bar['files']),per_domain={k:dict(v) for k,v in decisions.items()},cost=dict(cost),GT_read=False)
    write(OUT/'TEACHER_COUNTS.json',teacher)
    handoff=read(BASE/'ACTUAL_MEASUREMENT_HANDOFF.json')
    peak=max(read(ROOT/f['path'])['GPU_peak_bytes'] for f in bar['files'])
    write(OUT/'EXECUTION_COST.json',dict(status='complete',actual_new_teacher_generations=49,teacher_model_cold_load_count=1,teacher_token_and_request_sums=dict(cost),teacher_peak_GPU_bytes=peak,complete_finite_chain_seconds=handoff['elapsed_seconds'],student_stage_seconds=read(BASE/'STUDENT_GLOBAL_BARRIER.json')['wall_seconds'],actual_new_fit_count=0,actual_new_optimizer_steps=0,actual_new_student_complete_forward=0,original_top1_and_proposals_temporal_predictions_reused=True,raw_end_to_end_method_latency_remeasured=False,limitation='Measured new bounded phase only; includes one frozen teacher cold load, not historical proposal/Top1/Temporal costs or a fresh whole-method latency.',GT_read=False))
    root=read(BASE/'ROOT_STUDENT_READBACK.json')
    lines=['# Contrastive Referent Correction: fixed-pool development experiment','',
      'This is a finite development test on 32 previously exposed sources, 16 per domain. Frozen STVG-R1 compares the original Source and the original Top-1 non-Source choice using exactly the same two videos and nine-slot geometry. No new proposals, detectors, segmentation models, temporal feedback or hyperparameter search. This is not independent confirmation or Table 1.','',
      'All 128 original Top-1 slots are retained. Only 49 valid non-Source selections across 23 sources receive a pairwise request; Source, Uncertain and invalid replies add no reference. Alternative retains the exact original geometry. The candidate pool is fixed, so this verifier cannot introduce a correct object absent from that pool.','',
      '## Full Student predictions','', '| Arm | HC vIoU (%) | VidSTG vIoU (%) |','|---|---:|---:|']
    for r in rows:lines.append(f"| {r['arm']} | {r['HC_vIoU_percent']:.6f} | {r['Vid_vIoU_percent']:.6f} |")
    lines+=['','![Full-tube comparison](../results/stvg_contrastive_correction/2026-10-15/FULL_TUBE_RESULTS.png)','',f"Teacher counts: `{dict(teacher['per_domain'])}`.",'',
      f"New actual fitter invocations: {root['fit_invocations']}; accepted Adam steps from new invocations: {root['accepted_optimizer_steps']}. Exact historical fit/readout reuse and no-reference prediction reuse are counted separately in receipts. Arm-level totals can include historical reuse and are not all new execution. Gaussian IoU OPD, 1792 Query/LN parameters, query-based seed and per-domain configurations are fixed. This run has zero new fits, backward steps, source captures, or complete Student model redecodes because every Pair reference is empty. The 64 Pair arm identities are independently verified exact Frozen/Temporal reuse; no new OPD/decoder numerical qualification is claimed. Frozen/Temporal/historical predictions are exact reuse; no teacher tube substitutes the adapted Student.",'',
      'All 32 complete prediction records and independent source/reference/no-update identity checks precede GT opening. New sampled-reward and optimizer checks have denominator zero; this is not evidence that a new OPD update works. New GT is used only for post-seal evaluation, never reference construction, fitting, token repair or configuration selection. No bootstrap or oracle fitting. Supervised/unsupervised splits use actual Pairwise anchor support consistently within each source; reference diagnostic IoU is measured only at common real labeled anchor positions and cannot be numerically equated to full-tube vIoU. Unknown anchor positions are retained.','',
      '## Common observed reference diagnostics','', '| Domain | Retained | Total | Labeled | Positive | Negative | Neutral | Unknown | Mean IoU delta vs Source (pp) |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in diag:
        mean='unknown' if r['mean_delta_pp'] is None else f"{r['mean_delta_pp']:.6f}"
        lines.append(f"| {r['dataset']} | {r['retained']} | {r['count']} | {r['observed_count']} | {r['positive']} | {r['negative']} | {r['neutral']} | {r['unknown_count']} | {mean} |")
    lines += ['', '## Cost and execution', '', f"The finite teacher → reference lock → cached Student arm seal → independent readback → scoring chain took {handoff['elapsed_seconds']:.3f} seconds. It includes one R1 cold load and49 new calls; summed model generation/request times were {cost['generate_seconds']:.3f}/{cost['request_seconds']:.3f} seconds, with peak allocated GPU memory {peak} bytes. There was no new OPD fit or Student forward. Existing candidate generation, original Top-1 and Temporal costs are reused, so this is not raw end-to-end method latency or a Table1 estimate.", '', 'The initial CPU preflight rejected a self-referential job-file hash requirement before model launch. Original locks/source are preserved; dispatch002 removes only that impossible inside-file check, while the external runtime still pins the exact job-file SHA. AST readback confirms model loading, video input processing, generation and parsing code are unchanged. No OOM or resource-guard stop occurred. No new ID-swap/occlusion control was run; legal labels alone do not prove semantic discrimination.', '']
    decision=read(BASE/'TECHNICAL_DECISION.json')
    lines+=['','## Decision and limits','',decision['status'],'']+decision.get('reasons',[])+['',
      'Complete positive, negative, neutral and unknown source outcomes are available in the accompanying anonymous JSON. Propagation analysis of saved Delta-b trajectories is a separate CPU mechanism audit; teacher reference agreement is not semantic identity or GT correctness. Boundary-quality temporal adaptation and a new propagation constraint are not claimed as executed by this P0. CURRENT and the human-paused full run remain unchanged.','',
      '[Protocol](../protocols/stvg_contrastive_correction_v1.md) · [All anonymous outcomes](../results/stvg_contrastive_correction/2026-10-15).','']
    (ROOT/'docs/STVG_CONTRASTIVE_CORRECTION_V1.md').write_text('\n'.join(lines))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds,title in zip(axes,['hc2','vidstg'],['HC-STVG-v2','VidSTG']):
        arms=['TS_historical','TS_Pair'];v=[100*(lookup[ds,a,'m_vIoU']-lookup[ds,'T','m_vIoU']) for a in arms]
        bars=ax.bar(['Original Top-1','Pairwise'],v,color=['#8b99a6','#527da5']);ax.axhline(0,color='gray',lw=.8)
        for b,z in zip(bars,v):ax.annotate(f'{z:+.3f}',(b.get_x()+b.get_width()/2,z),ha='center',va='bottom' if z>=0 else 'top',xytext=(0,4 if z>=0 else -4),textcoords='offset points')
        ax.set_title(title+' (16 exposed sources)');ax.set_ylabel('Full-tube vIoU vs Temporal (pp)');ax.spines[['top','right']].set_visible(False);ax.margins(y=.25)
    fig.tight_layout();fig.savefig(OUT/'FULL_TUBE_RESULTS.png',dpi=170);plt.close(fig)
    print('CRC anonymous report prepared',len(rows))
if __name__=='__main__':main()
