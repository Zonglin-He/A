"""Anonymous complete-cohort Temporal T2/T4 export; no GT or model reader."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_t2_v1 import BASE,CORE,read,sha,put,verify
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.publish_stvg_soft_ep_v1 import verify_privacy
def main():
 import numpy as np
 from scipy.stats import t
 verify();done=read(BASE/'CPU_COMPLETION.json');assert done['status']=='complete' and done['count']==256 and done['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
 d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)};idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};checks=0
 assert len(d['rows'])==768 and set(idx)=={(r['probe_alias'],a) for r in rr for a in ('Frozen','T4','T2')}
 for s in d['summary']:
  ids=[r['probe_alias'] for r in rr if r['dataset']==s['dataset']];values=[idx[i,s['arm']][s['metric']] for i in ids if idx[i,s['arm']][s['metric']] is not None];assert len(values)==s['valid'] and s['unknown']==128-len(values)
  assert (s['mean'] is None and not values) or abs(math.fsum(values)/len(values)-s['mean'])<1e-12;checks+=1
 for c in d['paired']:
  a,b=c['comparison'].split('_minus_');ids=[r['probe_alias'] for r in rr if r['dataset']==c['dataset']];v={i:100*(idx[i,a][c['metric']]-idx[i,b][c['metric']]) for i in ids if idx[i,a][c['metric']] is not None and idx[i,b][c['metric']] is not None};n=len(v);assert v==c['per_source_delta_pp'] and n==c['valid'] and 128-n==c['unknown']
  if n:
   mean=math.fsum(v.values())/n;assert abs(mean-c['mean_delta_pp'])<1e-11
   if n>1:err=float(t.ppf(.975,n-1)*np.std(list(v.values()),ddof=1)/math.sqrt(n));assert np.allclose([mean-err,mean+err],c['paired_source_t95_interval_pp'],atol=1e-11,rtol=0)
  assert (c['positive'],c['negative'],c['neutral'])==(sum(x>1e-6 for x in v.values()),sum(x< -1e-6 for x in v.values()),sum(abs(x)<=1e-6 for x in v.values()));checks+=n
 for r in rr:
  for a in ('T4','T2'):assert idx[r['probe_alias'],a]['m_sIoU_dense_GT']==idx[r['probe_alias'],'Frozen']['m_sIoU_dense_GT']
  checks+=2
 def anon(x):
  if isinstance(x,list):return [anon(z) for z in x]
  if isinstance(x,dict):return {aliases.get(k,k):anon(v) for k,v in x.items() if k!='GT_bindings'}
  return aliases.get(x,x) if isinstance(x,str) else x
 out=ROOT/'results/stvg_temporal_t2_vs_t4_v1/2026-10-17/t2t4';out.mkdir(parents=True,exist_ok=False)
 def save(name,value):put(out/name,anon(value))
 save('RESULTS.json',d);save('METHOD_LOCK.json',read(BASE/'METHOD_LOCK.json'));save('CPU_CONTRACT_COMPLETION.json',read(BASE/'CPU_CONTRACT_COMPLETION.json'));save('QUALIFICATION_COMPLETION.json',read(BASE/'QUALIFICATION_COMPLETION.json'));save('ROOT_INPUT_PIXEL_VIEW_READBACK.json',read(BASE/'ROOT_INPUT_PIXEL_VIEW_READBACK.json'));save('ROOT_T4_SHARED_OPTIMIZER_READBACK.json',read(BASE/'ROOT_T4_SHARED_OPTIMIZER_READBACK.json'))
 teacher=read(BASE/'TEACHER_GLOBAL_BARRIER.json');old=read(CORE/'temporal/TEACHER_GLOBAL_BARRIER.json');stages=[]
 for p in sorted((BASE/'execution').glob('*/WORKER_EXIT.json')):
  v=read(p);assert v['exit_code']==0;stages.append(dict(stage=p.parent.name,wall_seconds=v['wall_seconds'],GT_read=v['GT_read']))
 cost=dict(count=256,extra_root_T4_CPU_replay_qualification_fits=2,extra_T2_full_fit_replay_count=256,peak_GPU_memory_GiB=None,peak_memory_status='unknown: headroom guard active but historical peak not retained',T2_logical_candidate_pairs=len(teacher['files']),T4_logical_candidate_pairs=len(old['files']),new_T2_Caption_alignment_pairs=teacher['new_caption_alignment_pairs'],new_T2_alignment_only=teacher['new_alignment_only_calls'],exact_old_pairs_reused=teacher['exact_reused_pairs'],extra_qualification_pairs=teacher['extra_qualification_pairs'],teacher_model_load_seconds=teacher['load_seconds'],teacher_wall_seconds=teacher['wall_seconds'],stages=stages,per_job=teacher['records'],per_fit=read(BASE/'STUDENT_GLOBAL_BARRIER.json')['records'],T4_historical_costs_reused_not_zero=True,fresh_end_to_end_speedup=None,fresh_speedup_status='unknown: actual cached T2 wall is measured, no same-process fresh T4 rerun; logical pair saving is not measured latency saving',GT_read=False);save('COST.json',cost)
 criteria=[]
 for ds in ('hc2','vidstg'):
  for key in ('m_tIoU','m_vIoU'):
   c=next(c for c in d['paired'] if c['dataset']==ds and c['comparison']=='T2_minus_T4' and c['metric']==key);ci=c['paired_source_t95_interval_pp'];criteria.append(dict(dataset=ds,metric=key,margin_pp=.2,paired95_t_interval_pp=ci,noninferiority_supported=ci is not None and ci[0]>=-.2))
 decision=dict(status='complete',count=256,effect_criteria=criteria,all_effect_criteria_met=all(x['noninferiority_supported'] for x in criteria),logical_pairs_reduced=cost['T2_logical_candidate_pairs']<cost['T4_logical_candidate_pairs'],matched_fresh_cost_criterion='unknown',automatic_promotion=False,CURRENT_changed=False,development_not_confirmation=True,GT_read=False);save('DECISION.json',decision)
 save('ROOT_SCALAR_READBACK.json',dict(status='pass',count=256,independent_checks=checks,complete_positive_negative_neutral_unknown=True,Frozen_spatial_trajectory_unchanged=True,no_new_model_or_GT_reader=True,GT_read=False))
 lines=['# Temporal two versus four candidates','', 'Fixed exposed development cohort: HC128 and VidSTG128, one independent parent/query per source. This is not a new confirmation. Frozen and T4 are exact sealed controls. Spatial adaptation is disabled in both Temporal arms.','', 'T2 retains Native and the highest-scoring different legal union of the original two frozen offset grids. Joint score is the sum of frozen grid scores, with original grid-row order for ties. Eight distinct legal physical frames are required; a missing distinct alternative produces an exact Native no-update. No Caption reward, GT, expansion ratio or contraction ratio selects the alternative. The original 512-parameter Temporal-Lite L-BFGS/Armijo, shared-mean candidate KL times offset count, Full-grid Trust, full-grid decode, original Caption/consistency Prompt, 8x448 input, corrected clock, greedy160-token cap and precision are preserved. Only candidate count and the predeclared independent two-candidate eligibility change. Original T4 gate and fallbacks remain untouched.','', '| Domain | Arm | vIoU % | tIoU % | dense sIoU % | R@0.3 % | R@0.5 % |','|---|---|---:|---:|---:|---:|---:|']
 def val(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
 for ds in ('hc2','vidstg'):
  for a in ('Frozen','T4','T2'):lines.append('|'+ds+'|'+a+'|'+ '|'.join('unknown' if (v:=val(ds,a,k)) is None else f'{100*v:.6f}' for k in ('m_vIoU','m_tIoU','m_sIoU_dense_GT','R_vIoU_at_0p3','R_vIoU_at_0p5'))+'|')
 lines+=['','| Domain | Contrast | Metric | Delta pp | Paired source 95% t interval pp | + / - / neutral / unknown |','|---|---|---|---:|---|---|']
 for c in d['paired']:
  if c['metric'] in ('m_tIoU','m_vIoU'):lines.append(f"|{c['dataset']}|{c['comparison']}|{c['metric']}|{c['mean_delta_pp']:+.6f}|{c['paired_source_t95_interval_pp']}|{c['positive']}/{c['negative']}/{c['neutral']}/{c['unknown']}|")
 lines+=['', 'The all256 table is primary. Pre-adaptation common eligibility is a separately reported diagnostic, with its own domain-specific sample counts; successful updates are never used to select the main cohort. No-information, no-fit and optimizer termination reasons are preserved for every source. Pairwise95% t intervals and signed outcomes include all evaluable sources, with unknowns retained. The prelocked tolerance is -0.2pp for both tIoU and vIoU in each domain; a failed or unresolved criterion does not trigger another search or automatic promotion.','', f"T2 has {cost['T2_logical_candidate_pairs']} logical candidate pairs versus T4 {cost['T4_logical_candidate_pairs']}. Actual added work is {cost['new_T2_Caption_alignment_pairs']} Caption/alignment pairs, {cost['new_T2_alignment_only']} alignment-only calls and {cost['extra_qualification_pairs']} extra original-adapter qualification pairs; {cost['exact_old_pairs_reused']} complete old pairs are reused only after exact Source/input pixels/sampling/FPS/own-query/complete Prompt checks. Separate real stage/per-job costs include cache handling and qualification. They are cached experiment wall times, not a same-process fresh T2/T4 speed benchmark; fresh end-to-end speedup remains unknown.", '', 'All predictions and exact full-fit replay are sealed before the new GT scoring pass. GT is absent from candidate selection, Caption rewards, optimization, acceptance and fallback. Existing T4 and failed B confirmation are preserved. No Spatial change, OPD resume, Prompt/LR/budget search, official fullset, CURRENT change or project FINAL. Private query/Caption text, Teacher raw responses, media, annotation/GT coordinates, tensors, weights and credentials are excluded.']
 doc=ROOT/'docs/STVG_TEMPORAL_T2_T4_001.md';doc.write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,ds in zip(axs,('hc2','vidstg')):
  cc=[next(c for c in d['paired'] if c['dataset']==ds and c['comparison']=='T2_minus_T4' and c['metric']==k) for k in ('m_vIoU','m_tIoU')];means=[c['mean_delta_pp'] for c in cc];err=[(c['paired_source_t95_interval_pp'][1]-c['paired_source_t95_interval_pp'][0])/2 for c in cc];ax.bar(['vIoU','tIoU'],means,yerr=err,capsize=4);ax.axhline(0,color='black',lw=.8);ax.axhline(-.2,color='red',ls='--',lw=.8);ax.set(title=ds,ylabel='T2 minus T4 (percentage points)')
 fig.tight_layout();fig.savefig(out/'temporal_effects.png',dpi=160);plt.close(fig)
 for name in ('stvg_temporal_t2_v1.py','continue_stvg_temporal_t2_v1.py','score_stvg_temporal_t2_v1.py','report_stvg_temporal_t2_v1.py','publish_stvg_temporal_t2_v1.py'):
  p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
 paths={str(doc.relative_to(ROOT))}|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()};verify_privacy(paths)
 for rel in paths:
  if (ROOT/rel).suffix=='.json':
   raw=(ROOT/rel).read_text()
   for cid in aliases:assert ('"'+cid+'"') not in raw,rel
 put(BASE/'publication/t2t4/EXPORT_MANIFEST.json',dict(status='complete',count=256,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('T2_T4_COMPLETE_ANONYMOUS_EXPORT_PENDING_ROOT_PIXELS_REMOTE')
if __name__=='__main__':main()
