import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_matched_ablation_a1_v1 import OUT,verify

def run():
 p=verify();a=read(OUT/'ACROSS_ORDERS.json');s=read(OUT/'SUMMARY.json');audit=read(OUT/'AUDIT.json');rows=read(OUT/'ROWS.json');diag=read(OUT/'DIAGNOSTICS.json')
 # Independent public scalar aggregation is separate from report generation.
 from scripts.audit_tastvg_matched_ablation_a1_public_v1 import run as public_audit
 check=public_audit(OUT);write(OUT/'PUBLIC_AUDIT.json',check)
 allocations=[read(f) for f in (OUT/'allocations').glob('*.json')];resource=dict(GPU_seconds=sum(x['seconds'] for x in allocations),failed_GPU_attempts=sum(x['status']!='completed' for x in allocations),new_arrivals=1440,new_spatial_updates=sum(x['updated'] for x in diag),new_spatial_probe_rollouts=2*120*9,new_expert_inferences=0,new_encoder_captures=0);write(OUT/'RESOURCES.json',resource)
 names={'Frozen':'Frozen','Fast-only':'Fast-only','Slow-only':'Slow-only / R-OPD','Final':'Final (frozen)','random_rank':'Random-Rank','off_policy':'Off-Policy','direct_pl':'Direct PL'}
 def fmt(arm,sub,key,group='corruption',pp=True):
  v=a[arm][group][sub][key];m=100 if pp else 1;return f"{v['mean']*m:+.6f} ± {v['sample_std']*m:.6f}"
 order=list(names);lines=['# Phase A: matched ablations of the frozen J0 recipe','', 'All three controls completed in one batch on the same16previously exposed VidSTG sources, five prelocked J0.1 orders, clean and five5%seed0conditions. Final method unchanged. No winner selected or configuration retuned.','', '| Method | Whole vIoU % | Whole Δv vsFrozen pp | Future Δs vsFast pp | Future Δv vsFast pp | Whole source >5pp v-harm count/16 | Future source >5pp v-harm count/12 |','|---|---:|---:|---:|---:|---:|---:|']
 for arm in order:lines.append('| '+names[arm]+' | '+' | '.join(fmt(arm,sub,key,pp=pp) for sub,key,pp in [('all','v',True),('all','delta_v',True),('nonexpert','delta_s',True),('nonexpert','delta_v',True),('all','source_harm_v',False),('nonexpert','source_harm_v',False)])+' |')
 lines+=['','Values are equal-order mean ± sampleSD across5orders; orders share sources, not independent datasets. Source corruption mean averages the five conditions first. Harm is paired source-mean delta<−5pp; per-cell harm and sIoU harm are separately retained in SUMMARY/ACROSS_ORDERS. Nonexpert Fast equals Frozen.','', '## Paired controls versus Final','', '| Control | Whole control−Final Δv pp | Future control−Final Δv pp | Future Δv positive/negative orders |','|---|---:|---:|---|']
 for arm in p['arms']:
  v=a[arm]['corruption']['nonexpert']['delta_v'];lines.append(f"| {names[arm]} | {fmt(arm,'all','minus_final_v')} | {fmt(arm,'nonexpert','minus_final_v')} | {v['positive']}/{v['negative']} |")
 lines+=['','## Every locked order','', '| Method/order | Whole Δv pp | Future Δs pp | Future Δv pp | Whole source v-harm | Future source v-harm |','|---|---:|---:|---:|---:|---:|']
 for arm in ['Final']+p['arms']:
  for o in p['orders']:
   x=s[arm][o]['corruption'];lines.append(f"| {names[arm]} / {o} | {x['all']['delta_v']*100:+.6f} | {x['nonexpert']['delta_s']*100:+.6f} | {x['nonexpert']['delta_v']*100:+.6f} | {x['all']['source_harm_v']} | {x['nonexpert']['source_harm_v']} |")
 lines+=['','## Clean control','', '| Method | Whole Δv pp | Future Δs pp | Future Δv pp |','|---|---:|---:|---:|']
 for arm in order:lines.append('| '+names[arm]+' | '+' | '.join(fmt(arm,sub,key,'clean') for sub,key in [('all','delta_v'),('nonexpert','delta_s'),('nonexpert','delta_v')])+' |')
 lines+=['','## Actual interventions and interpretation','', 'Random-Rank permutes rank assignment using a prelocked source/order hash and preserves the rank multiset and probability spectrum of its own current-policy support. Diverged states can lead to different later supports/rewards from Final; no claim that the complete trajectories retain equal update norms or q entropy.','', 'Off-Policy generates current-sample probes from fixed source parameters, but fits the current persistent actor against those detached boxes. Temporal candidates remain current-policy.','', 'Direct PL uses the same valid sparse Sa2VA masks,1792parameters,SGD.005 and pre-update output; mean frame loss is sum-coordinate L1+GIoU with coefficients1/1 as specified. No probes, norm matching or loss-scale tuning. This tests the fixed configuration, not all pseudo-label algorithms.','', 'Existing Raw/Rank/Rank+Norm results remain in S1.1 with their original single ordinal order. They are historical, not five-order matched rows. Existing critic/probe qualifications and J0/J0.1 schedule evidence are reused without new GPU reruns.','', f"Execution: {resource['GPU_seconds']:.3f}GPU-process seconds, {resource['new_spatial_updates']}updates,2160spatial probes,0new experts/encoder captures. {audit['state_links']}state links/{audit['state_resets']}resets, {audit['SGD_coordinates']}SGD coordinates, {audit['KL_updates']}rank KL and{audit['PL_updates']}PL losses, {audit['dual_metric_calls']}dual metrics,12full spatial/6six-layer temporal reinsertions,3CPU tests and{check['checks']}public scalar checks. Predictions sealed before re-reading16old labels.",'', 'Ablation outcomes constrain empirical claims, not frozen method selection. PhaseB remains the unchanged recipe; all same-source/condition/order dependence and historical exposure are retained.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 comparisons={arm:{sub:a[arm]['corruption'][sub]['minus_final_v'] for sub in ['all','nonexpert']} for arm in p['arms']};write(OUT/'DECISION.json',dict(status='completed',frozen_method_unchanged=True,control_minus_final=comparisons,no_winner_promotion=True,full_evaluation_next=True,time=time.time()))
 zh=['# Phase A：冻结方法的三项匹配消融','', '三臂全部实际运行，不按结果重选方法；沿用16曝光来源、五个J01预锁顺序和六条件。指标均为pp，mean±SD针对五序。','', '| 方法 | 全流Δv | Future Δs | Future Δv |','|---|---:|---:|---:|']
 for arm in order:zh.append('| '+names[arm]+' | '+' | '.join(fmt(arm,sub,key) for sub,key in [('all','delta_v'),('nonexpert','delta_s'),('nonexpert','delta_v')])+' |')
 zh+=['', 'Random只改rank assignment；Off只把spatial probes中心固定到source；PL按附件L1+GIoU系数1/1，只有效观察帧，未调scale。三者均保持原1792参数/一步SGD.005/persistence/current pre-update/temporal rerank。Raw/Norm旧单序结果不混进五序主表。完整逐序、来源损害/cell损害、clean及control−Final差值见REPORT和JSON。','', '所有对照正负结果保留，不以消融赢家替换冻结Final。新主评估依用户确认采用官方test全量，排除这一路开发来源；存在历史项目曝光，不称严格fresh。']
 (OUT/'RESEARCH_UPDATE.md').write_text('\n'.join(zh)+'\n')
 write(OUT/'PROVENANCE.json',dict(checkpoint_sha256=read(ROOT/'methods/tastvg_dual_evidence_j0_v1/config.json')['source_checkpoint_sha256'],lock_sha256=sha(OUT/'LOCK.json'),freeze_sha256=p['freeze_sha256'],pins=p['pins'],parent_prediction_barrier=p['J01_barrier_sha256'],sources=16,queries=16,orders=5,conditions=p['conditions']))
 print('\n'.join(zh))
if __name__=='__main__':run()
