"""Independent saved-scalar audit and anonymous geometry/config factorial export."""
from pathlib import Path
import math,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_geometry32_v2 import BASE,read,sha,put
GROUP='geometry32'
def main():
 import numpy as np
 from scipy.stats import t
 from scripts.export_stvg_lite_batch001 import public_source_text
 from scripts.publish_stvg_soft_ep_v1 import verify_privacy
 assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
 d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)};idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};checks=0
 def audit(values,p):
  nonlocal checks
  assert values==p['per_source_delta_pp'];n=len(values);assert n==p['valid'] and 16-n==p['unknown'];vs=list(values.values());mean=math.fsum(vs)/n if n else None;assert mean==p['mean_delta_pp'];ci=None
  if n>1:
   e=float(t.ppf(.975,n-1)*np.std(vs,ddof=1)/math.sqrt(n));ci=[mean-e,mean+e]
  assert (ci is None and p['paired_source_t95_interval_pp'] is None) or np.allclose(ci,p['paired_source_t95_interval_pp'],rtol=0,atol=1e-11)
  assert [sum(v>1e-6 for v in vs),sum(v< -1e-6 for v in vs),sum(abs(v)<=1e-6 for v in vs)]==[p[k] for k in ['positive','negative','neutral']];checks+=n
 for s in d['summary']:
  ids=[r['probe_alias'] for r in rr if r['dataset']==s['dataset']];vs=[idx[c,s['arm']][s['metric']] for c in ids if idx[c,s['arm']][s['metric']] is not None];assert len(vs)==s['valid'] and 16-len(vs)==s['unknown'];assert (not vs and s['mean'] is None) or abs(math.fsum(vs)/len(vs)-s['mean'])<1e-12;checks+=1
 for p in d['paired']:
  a,b=p['comparison'].split('_minus_');v={r['probe_alias']:100*(idx[r['probe_alias'],a][p['metric']]-idx[r['probe_alias'],b][p['metric']]) for r in rr if r['dataset']==p['dataset'] and idx[r['probe_alias'],a][p['metric']] is not None and idx[r['probe_alias'],b][p['metric']] is not None};audit(v,p)
 def anon(v):
  if isinstance(v,dict):return {aliases.get(k,k):anon(x) for k,x in v.items() if k!='GT_bindings'}
  if isinstance(v,list):return [anon(x) for x in v]
  
  if isinstance(v,str):
   if v in aliases:return aliases[v]
   for cid,alias in aliases.items():
    if v.startswith(cid+'_'):return alias+v[len(cid):]
  return v
 out=ROOT/'results/stvg_r1_geometry32/2026-10-17';out.mkdir(parents=True,exist_ok=False)
 for name,obj in [('RESULTS.json',d),('METHOD_LOCK.json',read(BASE/'METHOD_LOCK.json')),('REFERENCE_QUALITY.json',read(BASE/'POSTSEAL_REFERENCE_QUALITY.json')),('ROOT_READBACK.json',read(BASE/'ROOT_R1_GEOMETRY_READBACK.json')),('QUALIFICATION_COMPLETION.json',read(BASE/'R1_QUALIFICATION_COMPLETION.json')),('ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,full_signed_unknown_paired_audit=True,new_GT_reader=False))]:put(out/name,anon(obj))
 costs=[dict(stage=str(p.parent.relative_to(BASE)),**read(p)) for root in [BASE/'execution',BASE/'execution_measure'] for p in root.glob('**/WORKER_EXIT.json')];put(out/'COST.json',dict(stages=costs,R1_shared_geometry_generations=128,DINO_actual_calls=read(BASE/'DINO_GLOBAL_BARRIER.json')['actual_DINO_calls'],R1_geometry_direct_and_attention_share_prefill=True,new_Temporal=0,new_Source_models=0,old_baseline_cost_not_zero=True,fresh_end_to_end_cost=None))
 lines=['# Frozen R1 independent sparse geometry32','','Fixed exposed core first16 per domain,32parents/128original T2 anchors, unchanged Source/T2 and paper Spatial-Lite parameters/reset/response/acceptance. Current WF actual complete results are exact reuse. DIRECT and HEADS share the same unmarked context4, native-aspect separate anchor image and full Query, without DINO/Student geometry. DINO is an independent same-position original-interface baseline, acquired after the R1-only references were sealed. All three complete Student/ROOT/Global barriers precede GT. No head/Prompt/threshold/optimizer search.','','Single-sample LocalizationHeads adaptation follows official commit9ffe219d20ec376eb4dd14d42c54bb3299ffdb4a, default top5/chord/entropy.001/bottom-row.05/Gaussian1/above-mean all-positive bbox. This is not the paper dataset-statistical fixed-head main result. Actual attention is eager full-key causal softmax at the last complete-input text position, sliced only afterward to a separate anchor-image rectangular grid; no image-key re-softmax or merged-video-token frame alias. Direct outputs use actual processor-resized pixel width/height, invalid/multiple/out-of-range outputs remain failed/abstained, no normalized1000 conversion or clamp. Model namespace compatibility failure before first prefill and CPU pidfd waiter error are preserved with real extra cost; no science result changed. One native DINO boundary1.0000000298023224 failed a redundant exact check; recovery uses the already-existing1e-7 float32 interface tolerance, keeps every coordinate unchanged, resumes eight verified completed queries and only fits the remaining sources.','','| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|']
 def val(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
 arms=list(dict.fromkeys(r['arm'] for r in d['rows']))
 for ds in ['hc2','vidstg']:
  for a in arms:lines.append('|'+ds+'|'+a+'|'+'|'.join('unknown' if (v:=val(ds,a,k)) is None else f'{100*v:.6f}' for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
 lines+=['','| Domain | Contrast | vIoU delta pp | Paired source t95 pp | + / − / neutral / unknown |','|---|---|---:|---|---|']
 for p in d['paired']:
  if p['metric']=='m_vIoU' and any(p['comparison']==a+'_minus_'+b for a in ['DIRECT','HEADS','DINO'] for b in ['T2','Current_WF']):lines.append(f"|{p['dataset']}|{p['comparison']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
 lines+=['','Reference geometry/coverage/localization failures and unannotated unknowns are separately disclosed in REFERENCE_QUALITY.json; all128 planned positions remain. Available-reference mean IoU is conditional, while localization success rates use all GT-known planned positions including invalid/abstained outputs as failures. Box IoU and heatmaps do not establish identity/role correctness. Full source-paired deltas, observed/unobserved sIoU, signs/unknowns and update counts are in RESULTS.json. This is an exposed small development screen, not independent confirmation. Existing semantic512 panel remains44model-assisted/468pending. The practical stage2 decision is a prelocked resource decision, not production promotion or a universal impossibility claim. No CURRENT/paper change.','', 'Prelocked decision: '+json.dumps(read(BASE/'STAGE1_DECISION.json'))]
 doc=ROOT/'docs/STVG_R1_GEOMETRY32_001.md';doc.write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4));shown=['Current_WF','DIRECT','HEADS','DINO']
 for ax,ds in zip(axs,['hc2','vidstg']):ax.bar(shown,[100*(val(ds,a,'m_vIoU')-val(ds,'T2','m_vIoU')) for a in shown]);ax.axhline(0,color='black',lw=.7);ax.set_title(ds+' (all16 sources)');ax.set_ylabel('vIoU delta from T2 (pp)');ax.tick_params(axis='x',rotation=25)
 fig.tight_layout();fig.savefig(out/'effects.png',dpi=160);plt.close(fig)
 for name in ['stvg_r1_geometry32_v1.py','stvg_r1_geometry32_v2.py','stvg_r1_localization_readout_v1.py','stvg_r1_geometry32_measure_v1.py','run_stvg_r1_geometry32_v1.py','run_stvg_r1_geometry32_v2.py','run_stvg_r1_geometry32_measure_v1.py','run_stvg_r1_geometry32_measure_v2.py','quality_stvg_r1_geometry32_v1.py','report_stvg_r1_geometry32_v1.py','render_stvg_r1_geometry32_v1.py','publish_stvg_r1_geometry32_v1.py','resume_stvg_r1_geometry32_dino_v1.py','run_stvg_r1_geometry32_finish003.py']:
  p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
 paths={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_r1_geometry32_v1.md'};verify_privacy(paths);put(BASE/'publication'/GROUP/'EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('R1_GEOMETRY32_EXPORT_READY',checks)
if __name__=='__main__':main()
