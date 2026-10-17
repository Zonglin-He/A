"""Independent scalar audit; anonymous finite D0/D1/D2 report."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_dino_complement32_v1 import BASE,STAGE1,read,put,sha
def main():
 import numpy as np
 from scipy.stats import t
 from scripts.export_stvg_lite_batch001 import public_source_text
 from scripts.publish_stvg_soft_ep_v1 import verify_privacy
 assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json');d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)};idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};checks=0
 old={(r['case_id'],r['arm']):r['metrics'] for r in read(STAGE1/'POSTSEAL_RESULTS.json')['rows']}
 for r in rr:
  for a,b in [('Frozen','Frozen'),('T2','T2'),('Current_WF','Current_WF'),('D0','DINO')]:assert idx[r['probe_alias'],a]==old[r['probe_alias'],b];checks+=1
 for s in d['summary']:
  ids=[r['probe_alias'] for r in rr if r['dataset']==s['dataset']];v=[idx[c,s['arm']][s['metric']] for c in ids if idx[c,s['arm']][s['metric']] is not None];assert len(v)==s['valid'] and 16-len(v)==s['unknown'];assert (not v and s['mean'] is None) or abs(math.fsum(v)/len(v)-s['mean'])<1e-12;checks+=1
 for p in d['paired']:
  a,b=p['comparison'].split('_minus_');v={r['probe_alias']:100*(idx[r['probe_alias'],a][p['metric']]-idx[r['probe_alias'],b][p['metric']]) for r in rr if r['dataset']==p['dataset'] and idx[r['probe_alias'],a][p['metric']] is not None and idx[r['probe_alias'],b][p['metric']] is not None};assert v==p['per_source_delta_pp'];n=len(v);vs=list(v.values());mean=math.fsum(vs)/n if n else None;assert mean==p['mean_delta_pp'] and n==p['valid'] and 16-n==p['unknown'];ci=None
  if n>1:
   e=float(t.ppf(.975,n-1)*np.std(vs,ddof=1)/math.sqrt(n));ci=[mean-e,mean+e]
  assert (ci is None and p['paired_source_t95_interval_pp'] is None) or np.allclose(ci,p['paired_source_t95_interval_pp'],rtol=0,atol=1e-11);assert [sum(x>1e-6 for x in vs),sum(x< -1e-6 for x in vs),sum(abs(x)<=1e-6 for x in vs)]==[p[k] for k in ['positive','negative','neutral']];checks+=n
 def anon(v):
  if isinstance(v,dict):return {aliases.get(k,k):anon(x) for k,x in v.items() if k!='GT_bindings'}
  if isinstance(v,list):return [anon(x) for x in v]
  if isinstance(v,str):
   if v in aliases:return aliases[v]
   for cid,a in aliases.items():
    if v.startswith(cid+'_'):return a+v[len(cid):]
  return v
 out=ROOT/'results/stvg_r1_dino_complement32/2026-10-17';out.mkdir(parents=True,exist_ok=False);counts=[]
 for ds in ['hc2','vidstg']:
  decisions=[x for x in d['decisions'] if x['dataset']==ds];qs=[q for q in d['reference_quality'] if q['dataset']==ds and q['GT_known']];known_changed=[q for q in qs if all(q['references'][a]['available'] for a in ['D0','D1'])];sourcegroups=[]
  for r in [r for r in rr if r['dataset']==ds]:
   cc=[x for x in decisions if x['case_id']==r['probe_alias']];sourcegroups.append(dict(case_id=r['probe_alias'],has_no_pool=any(x['pool_count']==0 for x in cc),has_original_reject_pool=any(not x['D0_admitted'] and x['pool_count'] for x in cc),has_R1_abstention=any(x['status']=='abstained' for x in cc),has_alternative=any(x['selected_id'] not in [None,1] for x in cc)))
  counts.append(dict(dataset=ds,planned=len(decisions),no_pool=sum(x['pool_count']==0 for x in decisions),same_Top1=sum(x['selected_Top1'] for x in decisions),alternative=sum(x['selected_id'] not in [None,1] for x in decisions),abstained=sum(x['status']=='abstained' for x in decisions),invalid=sum(x['status'] in ['invalid_JSON','invalid_ID','truncated'] for x in decisions),original_admitted=sum(x['D0_admitted'] for x in decisions),selected_original_reject_pool=sum(not x['D0_admitted'] and x['selected_id'] is not None for x in decisions),GT_known=len(qs),GT_unknown=len(decisions)-len(qs),both_reference_known=len(known_changed),known_success_gained=sum(not q['references']['D0']['success_at_0p5'] and bool(q['references']['D1']['success_at_0p5']) for q in qs),known_success_lost=sum(bool(q['references']['D0']['success_at_0p5']) and not q['references']['D1']['success_at_0p5'] for q in qs),both_reference_IoU_improved=sum(q['references']['D1']['IoU']>q['references']['D0']['IoU']+1e-6 for q in known_changed),both_reference_IoU_worse=sum(q['references']['D1']['IoU']<q['references']['D0']['IoU']-1e-6 for q in known_changed),source_preGT_decision_groups=sourcegroups))
 for name,obj in [('RESULTS.json',d),('METHOD_LOCK.json',read(BASE/'METHOD_LOCK.json')),('SELECTION_READBACK.json',dict(status='complete',count=32,domains=counts,geometry_not_independent_identity_labels=True,abstention_strata_descriptive_not_causal=True)),('QUALIFICATION_COMPLETION.json',read(BASE/'QUALIFICATION_COMPLETION.json')),('ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,all_signed_unknown_pairs_checked=True,new_GT_reader=False))]:put(out/name,anon(obj))
 put(out/'COST.json',dict(stages=[dict(stage=str(p.parent.relative_to(BASE)),**read(p)) for p in (BASE/'execution').glob('**/WORKER_EXIT.json')],new_R1_generations=read(BASE/'TEACHER_GLOBAL_BARRIER.json')['new_R1_generations'],new_DINO=0,new_Temporal=0,D0_exact_reuse_old_cost_not_zero=True,D2_Student_updates=0,fresh_end_to_end_cost=None))
 lines=['# Same DINO geometry pool: original selection, R1 selection, and fixed propagation','','Same32 exposed parents/128 T2 observation anchors as the prior R1-only geometry screen. D0 is exact stage1 DINO original selection/admission plus unchanged paper OPD. D1 changes only the selector, using the same cached native DINO pool (context3 boxes, fallback10 boxes after the original SpatialExpert coordinate conversion) and original scores for stable ID order, native R1 full context4 plus one anchor-only contact sheet, one fixed Prompt and96-token cap. D2 uses exact D1 sparse references and Source trajectory, interpolating Source-relative center/log-size residuals in physical time within the same T2 interval; no Student parameter update, tracker or extra backbone. D1 retains the original paper episodic Gaussian/response/fullstep rollback/reject-stop; no score mix, optimizer redesign or Prompt/threshold search.','','The context interface supplies3 processed boxes per anchor; the original fallback supplies10 raw top boxes, converted with the exact original SpatialExpert float32 xyxy clamp/roundtrip before both selectors. Its admitted Top1 equals sealed D0 bitwise. Three CPU-only schema/conversion preparation failures before any new model/GT are preserved; no pool was chosen by GT. D0 original rejects and D1 selections from the same existing pool can change adaptation opportunity; these are separately counted and not presented as pure selection quality. No-pool positions remain missing. ID1 is original Top1 where admitted; fixed order may bias R1. Localization IoU success changes are not independent identity/role labels. Abstention subgroups are descriptive: mixed-window updates do not identify a causal abstention effect. All32 parents, failures, unknowns and negative tails remain in the paired report. D0/Frozen/T2/current WF are exact old predictions; their cost is not zero. New D1/D2 predictions and roots are sealed before any stage2 GT. No production promotion, confirmation or official fullset.','','| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|']
 def value(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
 for ds in ['hc2','vidstg']:
  for a in ['Frozen','T2','Current_WF','D0','D1','D2']:lines.append('|'+ds+'|'+a+'|'+'|'.join('unknown' if (v:=value(ds,a,k)) is None else f'{100*v:.6f}' for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
 lines+=['','| Domain | Contrast | vIoU delta pp | Paired source t95 pp | + / − / neutral / unknown |','|---|---|---:|---|---|']
 for p in d['paired']:
  if p['metric']=='m_vIoU' and p['comparison'] in ['D1_minus_D0','D1_minus_D2','D2_minus_D0','D1_minus_Current_WF','D0_minus_Current_WF']:lines.append(f"|{p['dataset']}|{p['comparison']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
 lines+=['','All metrics including nonanchor sIoU and every source-paired signed delta are in RESULTS.json. Reference selection/correctness proxies, missing-pool/admission differences and descriptive abstention groups are in SELECTION_READBACK.json. No semantic labels are inferred from these proxies; existing semantic512 panel remains44model-assisted/468pending. This finite complement study does not automatically authorize new modules.']
 doc=ROOT/'docs/STVG_R1_DINO_COMPLEMENT32_001.md';doc.write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,ds in zip(axs,['hc2','vidstg']):
  arms=['Current_WF','D0','D1','D2'];ax.bar(arms,[100*(value(ds,a,'m_vIoU')-value(ds,'T2','m_vIoU')) for a in arms]);ax.axhline(0,color='black',lw=.7);ax.set_title(ds+' (all16 parents)');ax.set_ylabel('vIoU delta from T2 (pp)')
 fig.tight_layout();fig.savefig(out/'effects.png',dpi=160);plt.close(fig)
 for name in ['retire_unused_storage_20261017_v2.py','stvg_r1_dino_complement32_v1.py','run_stvg_r1_dino_complement32_v1.py','report_stvg_r1_dino_complement32_v1.py','render_stvg_r1_dino_complement32_v1.py','publish_stvg_r1_dino_complement32_v1.py']:
  p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
 put(out/'STORAGE_CLEANUP.json',read(ROOT/'artifacts/storage_cleanup_20261017_unused_v2/COMPLETION.json'))
 paths={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'docs/STORAGE_CLEANUP_20261017_UNUSED_V2.md'};verify_privacy(paths);put(BASE/'publication/complement32/EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('COMPLEMENT32_EXPORT_READY',checks)
if __name__=='__main__':main()
