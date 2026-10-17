"""Independent saved-scalar audit and anonymous geometry/config factorial export."""
from pathlib import Path
import math,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_dino_pool_r1_32_v3 import BASE,read,sha,put
GROUP='selection32'
def main():
 import numpy as np
 from scipy.stats import t
 from scripts.export_stvg_lite_batch001 import public_source_text
 from scripts.publish_stvg_soft_ep_v1 import verify_privacy
 assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
 d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['case_id']:f'R{i:03d}' for i,r in enumerate(rr,1)};idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};checks=0
 old=read(ROOT/'artifacts/stvg_geometry_config32_v1/POSTSEAL_RESULTS.json');oldidx={(r['case_id'],r['arm']):r['metrics'] for r in old['rows']}
 for cid in aliases:
  for name in ['F','T']:
   assert idx[cid,name]==oldidx[cid,name]
  for pre in ['S_','TS_']:
   for suffix in ['SHORT','ORIGINAL']:assert idx[cid,pre+'Top1_'+suffix]==oldidx[cid,pre+'DINO_common_'+suffix]
 def audit(values,p):
  nonlocal checks
  assert values==p['per_source_delta_pp'];n=len(values);assert n==p['valid'] and 16-n==p['unknown'];vs=list(values.values());mean=math.fsum(vs)/n if n else None;assert mean==p['mean_delta_pp'];ci=None
  if n>1:
   e=float(t.ppf(.975,n-1)*np.std(vs,ddof=1)/math.sqrt(n));ci=[mean-e,mean+e]
  assert (ci is None and p['paired_source_t95_interval_pp'] is None) or np.allclose(ci,p['paired_source_t95_interval_pp'],rtol=0,atol=1e-11)
  assert [sum(v>1e-6 for v in vs),sum(v< -1e-6 for v in vs),sum(abs(v)<=1e-6 for v in vs)]==[p[k] for k in ['positive','negative','neutral']];checks+=n
 for s in d['summary']:
  ids=[r['case_id'] for r in rr if r['dataset']==s['dataset']];vs=[idx[c,s['arm']][s['metric']] for c in ids if idx[c,s['arm']][s['metric']] is not None];assert len(vs)==s['valid'] and 16-len(vs)==s['unknown'];assert (not vs and s['mean'] is None) or abs(math.fsum(vs)/len(vs)-s['mean'])<1e-12;checks+=1
 for p in d['paired']:
  a,b=p['comparison'].split('_minus_');v={r['case_id']:100*(idx[r['case_id'],a][p['metric']]-idx[r['case_id'],b][p['metric']]) for r in rr if r['dataset']==p['dataset'] and idx[r['case_id'],a][p['metric']] is not None and idx[r['case_id'],b][p['metric']] is not None};audit(v,p)
 def anon(v):
  if isinstance(v,dict):return {aliases.get(k,k):anon(x) for k,x in v.items() if k!='GT_bindings'}
  if isinstance(v,list):return [anon(x) for x in v]
  
  if isinstance(v,str):
   if v in aliases:return aliases[v]
   for cid,alias in aliases.items():
    if v.startswith(cid+'_'):return alias+v[len(cid):]
  return v
 out=ROOT/'results/stvg_dino_pool_r1_32/2026-10-17';out.mkdir(parents=True,exist_ok=False)
 for name,obj in [('RESULTS.json',d),('METHOD_LOCK.json',read(BASE/'METHOD_LOCK003.json')),('REFERENCE_QUALITY.json',read(BASE/'POSTSEAL_REFERENCE_QUALITY.json')),('ROOT_READBACK.json',read(BASE/'ROOT_READBACK.json')),('QUALIFICATION_COMPLETION.json',read(BASE/'QUALIFICATION_COMPLETION.json')),('ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,full_signed_unknown_paired_audit=True,new_GT_reader=False))]:put(out/name,anon(obj))
 costs=[dict(stage=p.parent.name,**read(p)) for p in (BASE/'execution003').glob('*/WORKER_EXIT.json')];assert all(r['exit_code']==0 for r in costs);bar=read(BASE/'STUDENT_GLOBAL_BARRIER.json');put(out/'COST.json',dict(stages=costs,new_scientific_fits=bar['new_scientific_fits'],exact_Top1_whole_Query_reuses=bar['exact_Top1_whole_Query_reuses'],Source_capture_calls=bar['source_capture_calls'],actual_full_RGB_qualification_calls=read(BASE/'STUDENT_QUALIFICATION_COMPLETION.json')['actual_full_RGB_qualification_calls'],new_WHO=38,extra_short_generations=6,new_DINO=0,prior_attempts=dict(A_partial_WHO=5,A_short_qualification=3,A_encoder_calls=6,A_worker_wall_seconds=read(BASE/'execution/teacher/WORKER_EXIT.json')['wall_seconds'],native_receipt_collision_short_qualification=3,native_receipt_collision_encoder_calls=1,native_receipt_collision_wall_seconds=read(BASE/'execution002/teacher/WORKER_EXIT.json')['wall_seconds']),Top1_cells='exact sealed historical reuse; historical cost is not zero',input_preparation_wall_seconds=None,fresh_end_to_end_cost=None))
 from scripts.stvg_reference_recovery_fit_v1 import locked_config
 put(out/'CONFIGS.json',{ds:{p:locked_config(ds,p) for p in ['current5','original']} for ds in ['hc2','vidstg']})
 lines=['# Same cached DINO pool: Top1 versus native R1 selector32','','Historical exposed HC16/Vid16;38 fixed common positions/19 supported sources,13 unsupported no-op sources retained. Exact cached processed_probe three-candidate pool per position and entity target scores. Top1 is ID1, then fixed descending target score/expert ID; possible position bias is disclosed. No new DINO. R1 sees native full1280 visual tokens, old crop Prompt/context4/local5 and original physical uniform-clock adapter; it is not compressed A or current B. Caption-independent, one fixed selector, no Prompt/Token search. All local pixels match historical SHA, center RGB also matches actual DINO input. Existing KLT only renders adjacent crops; actual center targets remain exact pool members.','','The first A160 attempt completed5WHO/3short qualification calls/6encoders, then stopped on its sixth job: valid real crop cells were insufficient for its fixed quota. Those partial A decisions are not mixed into the native comparison. A second native attempt stopped on an old receipt collision after3short calls/1encoder and0WHO. Failure/pins/logs are retained and directories isolated; native1280 avoids inventing observations or a new Token allocation rule. All extra cost and the final6short qualification calls are separately counted. Earlier zero-WHO wording was corrected.','','Same episodic Source/Adam/query RNG,1792 parameters,32Gaussian samples and original sparse-reference acceptance/rollback/reject-stop under both historical current5/original packages. Top1 cells exact whole-query reuse; identical entire R1 references also reuse Top1 exactly. Different/new R1 references are actually fit and full outputs recomputed. Invalid/abstained R1 references are omitted, not replaced with GT or Top1; all sources stay in the denominator. All predictions/ROOT/Global barriers preceded GT.','','| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|']
 def val(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
 arms=list(dict.fromkeys(r['arm'] for r in d['rows']))
 for ds in ['hc2','vidstg']:
  for a in arms:lines.append('|'+ds+'|'+a+'|'+'|'.join('unknown' if (v:=val(ds,a,k)) is None else f'{100*v:.6f}' for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
 lines+=['','| Domain | Fixed-package contrast | vIoU delta pp | Paired source t95 pp | + / − / neutral / unknown |','|---|---|---:|---|---|']
 for p in d['paired']:
  if p['metric']=='m_vIoU' and p['comparison'] in ['TS_R1_SHORT_minus_TS_Top1_SHORT','TS_R1_ORIGINAL_minus_TS_Top1_ORIGINAL','S_R1_SHORT_minus_S_Top1_SHORT','S_R1_ORIGINAL_minus_S_Top1_ORIGINAL']:
   lines.append(f"|{p['dataset']}|{p['comparison']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
 lines+=['','Selector decisions (all38, including invalid/0) and per-source update/reuse counts are preserved in RESULTS.json. Identity/role correctness cannot be inferred solely from box IoU or vIoU; no independent human semantic annotation is asserted.']
 lines+=['','All pairwise source deltas, signs, unknowns and intervals for all metrics are in RESULTS.json; t95 is descriptive on this exposed small development sample. The support is conditional on historical DINO admission. This tests the additional effect of a fixed native R1 selector on the same expert candidate pool within each configuration package, not current-B superiority or long-stream online state. Sparse-reference acceptance is not GT correctness. Post-seal observed-position reference IoU and selection coverage are in REFERENCE_QUALITY.json; missing GT and abstention remain distinct, with no identity/role-label claim. Actual pixel review is engineering inspection, not completion of the512-window semantic panel. Production CURRENT and frozen paper registry remain unchanged.']
 doc=ROOT/'docs/STVG_DINO_POOL_R1_32_001.md';doc.write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4));shown=['TS_Top1_SHORT','TS_R1_SHORT','TS_Top1_ORIGINAL','TS_R1_ORIGINAL']
 for ax,ds in zip(axs,['hc2','vidstg']):ax.bar(['Top1 short','R1 short','Top1 original','R1 original'],[100*(val(ds,a,'m_vIoU')-val(ds,'T','m_vIoU')) for a in shown]);ax.axhline(0,color='black',lw=.7);ax.set_title(ds+' (all16 sources)');ax.set_ylabel('vIoU delta from T (pp)');ax.tick_params(axis='x',rotation=25)
 fig.tight_layout();fig.savefig(out/'effects.png',dpi=160);plt.close(fig)
 for name in ['stvg_dino_pool_r1_32_v1.py','stvg_dino_pool_r1_32_v2.py','stvg_dino_pool_r1_32_v3.py','run_stvg_dino_pool_r1_32_v1.py','run_stvg_dino_pool_r1_32_v2.py','run_stvg_dino_pool_r1_32_v3.py','report_stvg_dino_pool_r1_32_v1.py','quality_stvg_dino_pool_r1_32_v1.py','render_stvg_dino_pool_r1_32_v1.py','publish_stvg_dino_pool_r1_32_v1.py']:
  p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
 paths={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_dino_pool_r1_32_v1.md'};verify_privacy(paths);put(BASE/'publication'/GROUP/'EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('DINO_POOL_R1_32_EXPORT_READY',checks)
if __name__=='__main__':main()
