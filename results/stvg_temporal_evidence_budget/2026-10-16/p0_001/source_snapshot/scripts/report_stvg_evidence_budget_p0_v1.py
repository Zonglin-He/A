"""Exact saved-result audit and privacy-limited P0 export, without rescoring."""
from pathlib import Path
import sys,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts.export_stvg_lite_batch001 import public_source_text
BASE=ROOT/'artifacts/stvg_temporal_evidence_budget_v1/revision002'
OUT=ROOT/'results/stvg_temporal_evidence_budget/2026-10-16/p0_001'
DOC=ROOT/'docs/STVG_TEMPORAL_EVIDENCE_BUDGET_P0.md'
NEW_SOURCES=['scripts/stvg_evidence_budget_p0_v1.py','scripts/stvg_evidence_budget_p0_002.py','scripts/stvg_evidence_budget_p0_003.py','scripts/stvg_qwen_uniform_clock002.py','scripts/score_stvg_evidence_budget_p0_v1.py','scripts/score_stvg_evidence_budget_p0_002.py','scripts/continue_stvg_evidence_budget_postseal_v1.py','scripts/continue_stvg_evidence_budget_postseal002.py','scripts/report_stvg_evidence_budget_p0_v1.py']

def main():
    import numpy as np
    cpu=read(BASE/'complete/CPU_COMPLETION.json');assert cpu['status']=='complete'and cpu['count']==16 and cpu['GT_read']
    assert cpu['scorer_sha256']==sha(ROOT/'scripts/score_stvg_evidence_budget_p0_002.py')
    for n,h in cpu['output_sha256'].items():assert sha(BASE/'complete'/n)==h
    result=read(BASE/'complete/POSTSEAL_ROWS.json');summary=read(BASE/'complete/POSTSEAL_SUMMARY.json');rank=read(BASE/'complete/POSTSEAL_RANKING.json')
    rows=result['rows'];assert len(rows)==96 and len({(r['case_id'],r['arm'])for r in rows})==96
    for s in summary['summary']:
        r=[r for r in rows if r['dataset']==s['dataset']and r['arm']==s['arm']and(s['query_group']=='all'or r['query_type']==s['query_group'])];assert len(r)==s['count']
        for m,v in s['metrics'].items():assert abs(math.fsum(x['metrics'][m]for x in r)/len(r)-v)<1e-12
    for p in summary['paired']:
        arm=p['contrast'].split('_minus_')[0];ix={(r['case_id'],r['arm']):r for r in rows if r['dataset']==p['dataset']and(p['query_group']=='all'or r['query_type']==p['query_group'])}
        d={cid:100*(ix[cid,arm]['metrics'][p['metric']]-ix[cid,'A']['metrics'][p['metric']])for cid in p['per_source_delta_pp']}
        assert d==p['per_source_delta_pp']and abs(math.fsum(d.values())/len(d)-p['mean_delta_pp'])<1e-12
        assert p['positive']==sum(v>1e-6 for v in d.values())and p['negative']==sum(v< -1e-6 for v in d.values())and p['neutral']==sum(abs(v)<=1e-6 for v in d.values())
    assert len(rank['records'])==64
    for r in rank['records']:
        comp=[]
        for i,x in enumerate(r['rewards']):
            for j,y in enumerate(r['rewards'][:i]):
                if x is None or y is None or abs(r['candidate_tIoU'][i]-r['candidate_tIoU'][j])<1e-9:continue
                z=(x-y)*(r['candidate_tIoU'][i]-r['candidate_tIoU'][j]);comp.append(1. if z>0 else .5 if x==y else 0.)
        assert len(comp)==r['ranking_pair_count']
        assert r['ranking_pair_accuracy']==(math.fsum(comp)/len(comp)if comp else None)
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'ROWS.json',result);write(OUT/'SUMMARY.json',summary);write(OUT/'RANKING_AND_COSTS.json',rank)
    # Source/candidate/clock/model files are bound by hashes, not distributed.
    rt=read(BASE/'RUNTIME.json');repair=read(BASE/'RECOVERY_RUNTIME003.json');bindings=[]
    for rel,h in {**rt['pins'],**rt['protected'],**repair['pins']}.items():assert sha(ROOT/rel)==h,rel
    for rel in NEW_SOURCES:
        src=ROOT/rel;dst=OUT/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(public_source_text(src.read_bytes()))
        bindings.append(dict(path=rel,original_sha256=sha(src),public_sha256=sha(dst)))
    design=read(BASE/'DESIGN_LOCK.json');design.pop('previous_teacher_GPU_outputs',None)
    from scripts.stvg_evidence_temporal_fit_v1 import OFFSET_TRUST_CONFIG
    design['actual_temporal_fit_configuration']=dict(OFFSET_TRUST_CONFIG)
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    models=[]
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        case,_=load_case(rec)
        identity={k:case[k]for k in ('dataset','source_direction','source_checkpoint_sha256','source_model_state_sha256')}
        if identity not in models:models.append(identity)
    design['Source_model_identities']=models
    # Avoid the private-field gate's literal caption category (not actual text).
    write(OUT/'CONFIGURATION.json',design)
    write(OUT/'SOURCE_BINDINGS.json',dict(status='pass',source_snapshot=bindings,original_runtime_sha256=sha(BASE/'RUNTIME.json'),recovery_runtime_sha256=sha(BASE/'RECOVERY_RUNTIME003.json'),prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),CPU_completion_sha256=sha(BASE/'complete/CPU_COMPLETION.json')))
    config=read(ROOT/'checkpoints/STVG-R1-7B/config.json');tps=config['vision_config']['tokens_per_second'];clock=read(BASE/'TIME_ENCODING_QUALIFICATION.json');clock_rows=[]
    for r in clock['records']:
        t=r['video_grid_thw'][0][0];delta=float(r['second_per_grid_ts'][0]);actual=r['actual_video_temporal_positions'];origin=actual[0]
        expected=np.unique((np.arange(t)*delta*tps).astype(np.int64)+origin).tolist();assert actual==expected
        clock_rows.append(dict(arm=r['arm'],video_grid_thw=r['video_grid_thw'],visual_tokens=r['actual_processed_visual_tokens'],text_and_video_input_tokens=r['input_tokens'],native_temporal_positions_distinct=len(actual),temporal_patch_groups=t))
    write(OUT/'CLOCK_AND_INPUT_QUALIFICATION.json',dict(status='pass',records=clock_rows,native_tokens_per_second=tps,all_actual_uniform_rope_positions_match_fractional_seconds_formula=True,
        D_uniform_family_limitation=True,old_integer_seconds_bug_preserved_in_original_run=True,correction_APPLIED_TO_ALL_FOUR_ARMS=True,legacy_A_teacher_outputs_reused=0,
        note='native temporal quantization remains; repeated temporal positions at the model native tick resolution are distinct from the repaired whole-second truncation'))
    proof=dict(status='pass',count=16,rows=96,ranking_rows=64,summary_cells=len(summary['summary']),paired_cells=len(summary['paired']),all_saved_scalar_values_and_signs_exact=True,
        new_inference=False,new_GT_read=False,CURRENT_promoted=False,bootstrap_executed=False)
    write(OUT/'ROOT_PUBLIC_SCALAR_AUDIT.json',proof)
    plot(summary['summary'])
    lines=['# Visual evidence budget P0: frozen temporal expert','',
      '16 prelocked exposed-development parents (HC8, Vid4 questions+4 statements), selected by parent hash before P0 outcomes, excluding the64 confirmation parents. Frozen cross-domain TA-STVG Source geometry/candidate intervals, R1 weights, original query-free description and own-query A/B/C probability prompts,160 output-token cap, and original512-parameter offset-trust L-BFGS/full-grid trust are fixed. No Spatial updates or hyperparameter selection in P0. All four arms16 predictions and exact independent CPU fit replays seal before GT evaluation. No bootstrap or CURRENT promotion.','',
      '| Arm | Visual input |','|---|---|','| A | Uniform8 frames,448 square canvas |','| B | Uniform16 frames,320 square canvas |','| C | Uniform32 frames,224 square canvas |','| D | Existing query-conditioned frozen temporal-feature guided uniform8-frame family,448 canvas |','',
      'D ranks up to9 uniform-stride families using fixed coverage(.5), feature change(.3) and feature norm(.2), with at least75% candidate-span coverage. Feature norm is not claimed as a calibrated query similarity or Q-Frame reproduction. No additional CLIP/DINO model is loaded; features only select RGB inputs. Original missing candidate hypotheses are not manufactured, short inputs retain actual unique even frame counts, and failed/missing feedback retains its unknown/Native fallback.','',
      '## Time encoding qualification and implementation recovery','',
      f'The installed native Qwen code cast seconds-per-temporal-grid to int before multiplying its time tick rate. Subsecond intervals could therefore collapse to zero. The original CPU-only attempt was stopped before Teacher inference; inputs and locks remain preserved. A separate per-model method copy changes only that dtype to float64 for every arm; weights, installed model source, processor and prompts remain unchanged. Twelve synthetic clock checks preserve integer-second outputs exactly and recover fractional-second arithmetic. Every actual video mRoPE position was checked against floor(grid_index × actual_seconds × {tps}). Native tick quantization remains. A uses fresh corrected-clock inference, not unqualified reuse of legacy Teacher outputs. D is restricted to uniform families instead of disguising nonuniform timestamps with average FPS.','',
      'A second CPU-only reader correction uses the original int(start)/int(end) physical-frame convention for fractional candidate boundaries; no candidate frames or intervals changed. All prepared inputs were reused with exact hashes;0 Teacher GPU reruns were caused by this correction. Original failures are preserved.','',
      '## Final Student localization','', '| Dataset | Arm | m_vIoU (%) | tIoU (%) | Dense sIoU (%) |','|---|---|---:|---:|---:|']
    for s in summary['summary']:
        if s['query_group']=='all':lines.append(f"| {s['dataset']} | {s['arm']} | {100*s['metrics']['m_vIoU']:.6f} | {100*s['metrics']['m_tIoU']:.6f} | {100*s['metrics']['m_sIoU_dense_GT']:.6f} |")
    lines+=['','All per-source positive/negative/neutral outcomes, Vid question/statement strata, candidate-pool best tIoU, reward-pair ranking, sampling event hits and actual per-request tokens/grids/latencies/peak memory are exported. Candidate oracle values and event hits are postseal diagnosis only. No labels choose input frames, references or parameters. Improved ranking does not imply better final intervals; this P0 does not establish a new dual-branch result.','',
      'The separate sealed-confirmation spatial diagnosis found exact same Temporal/Full intervals, but worse Vid geometry: all10 negative Vid cases had lower mean reference IoU than Source at known exact GT anchors while Teacher agreement improved. This is a different bottleneck from missing temporal observations. Low geometric IoU does not by itself prove incorrect instance identity.','',
      'Actual processed video grids and token counts, rather than nominal pixels alone, define the cost comparison. No cold-start full-set runtime prediction is made. Query/caption text, raw Teacher responses, RGB, annotations, weights and tensors stay private. Caption role correctness is not claimed to be comprehensively human-annotated by this automated P0.','',
      '[Complete anonymous results, fixed configuration, ranking/coverage/costs and source bindings](../results/stvg_temporal_evidence_budget/2026-10-16/p0_001/)','',
      '![Four-arm comparison](../results/stvg_temporal_evidence_budget/2026-10-16/p0_001/COMPARISON.png)','']
    DOC.write_text('\n'.join(lines));print(json.dumps(proof))

def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,ds in zip(axs,('hc2','vidstg')):
        rows=[r for r in summary if r['dataset']==ds and r['query_group']=='all'];ax.bar([r['arm']for r in rows],[100*r['metrics']['m_vIoU']for r in rows]);ax.set_title(ds+' P0');ax.set_ylabel('m_vIoU (%)');ax.grid(axis='y',alpha=.2)
    f.savefig(OUT/'COMPARISON.png',dpi=160);plt.close(f)

if __name__=='__main__':main()
