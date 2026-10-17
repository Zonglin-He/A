"""Postseal scalar audit, anonymous public export; no model or new GT reader."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_followups64_v1 import BASE,read,sha,put,verify
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.publish_stvg_soft_ep_v1 import verify_privacy

def main(family):
 import numpy as np,torch
 from scipy.stats import t
 verify();b=BASE/family;done=read(b/'CPU_COMPLETION.json');assert done['status']=='complete' and done['count']==64 and done['result_sha256']==sha(b/'POSTSEAL_RESULTS.json');assert read(b/'ROOT_READBACK.json')['status']=='pass'
 d=read(b/'POSTSEAL_RESULTS.json');roster=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(roster,1)};idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};arms=list(dict.fromkeys(r['arm'] for r in d['rows']));checks=0
 assert len(idx)==len(d['rows'])==64*len(arms)
 for s in d['summary']:
  ids=[r['probe_alias'] for r in roster if r['dataset']==s['dataset']];vals=[idx[i,s['arm']][s['metric']] for i in ids if idx[i,s['arm']][s['metric']] is not None];assert len(vals)==s['valid'] and 32-len(vals)==s['unknown'];assert (not vals and s['mean'] is None) or abs(math.fsum(vals)/len(vals)-s['mean'])<1e-12;checks+=1
 for c in d['paired']:
  a,z=c['comparison'].split('_minus_');ids=[r['probe_alias'] for r in roster if r['dataset']==c['dataset']];v={i:100*(idx[i,a][c['metric']]-idx[i,z][c['metric']]) for i in ids if idx[i,a][c['metric']] is not None and idx[i,z][c['metric']] is not None};n=len(v);assert v==c['per_source_delta_pp'] and n==c['valid'] and 32-n==c['unknown'];mean=math.fsum(v.values())/n if n else None;assert mean==c['mean_delta_pp'];err=float(t.ppf(.975,n-1)*np.std(list(v.values()),ddof=1)/math.sqrt(n)) if n>1 else None;assert (err is None and c['paired_source_t95_interval_pp'] is None) or np.allclose([mean-err,mean+err],c['paired_source_t95_interval_pp'],atol=1e-11,rtol=0);assert (c['positive'],c['negative'],c['neutral'])==(sum(x>1e-6 for x in v.values()),sum(x< -1e-6 for x in v.values()),sum(abs(x)<=1e-6 for x in v.values()));checks+=n
 for r in roster:
  v=[idx[r['probe_alias'],a]['m_tIoU'] for a in arms];assert all(x==v[0] for x in v);checks+=len(v)
 def anon(x):
  if isinstance(x,list):return [anon(z) for z in x]
  if isinstance(x,dict):return {aliases.get(k,k):anon(v) for k,v in x.items() if k!='GT_bindings'}
  return aliases.get(x,x) if isinstance(x,str) else x
 out=ROOT/'results/stvg_opd_followups64_v1/2026-10-17'/family;out.mkdir(parents=True,exist_ok=False)
 def save(n,x):put(out/n,anon(x))
 for n,p in [('RESULTS.json',b/'POSTSEAL_RESULTS.json'),('METHOD_LOCK.json',BASE/'METHOD_LOCK.json'),('CPU_CONTRACT_COMPLETION.json',BASE/'CPU_CONTRACT_COMPLETION.json'),('ROOT_READBACK.json',b/'ROOT_READBACK.json'),('STAGE_DIAGNOSTICS.json',b/'ROOT_STAGE_DIAGNOSTICS.json')]:save(n,read(p))
 if family=='anneal':save('QUALIFICATION_COMPLETION.json',read(b/'QUALIFICATION_COMPLETION.json'))
 else:save('ROOT_COMPLETION_MATH_READBACK.json',read(b/'ROOT_COMPLETION_MATH_READBACK.json'))
 configs={};noops=[]
 for f in read(b/'STUDENT_GLOBAL_BARRIER.json')['files']:
  p=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);noops.append(dict(case_id=p['case_id'],dataset=p['dataset'],no_reference=p['no_reference_noop'],invalid_chart=p['chart_noop'],actual_fit_count=p['actual_fit_count']))
  for arm,fit in p['fits'].items():
   if fit is not None and arm not in configs.setdefault(p['dataset'],{}):configs[p['dataset']][arm]=dict(config=fit['config'],settings=fit['settings'],scale_schedule=fit.get('scale_schedule'),noise_slots=fit.get('noise_slots'),active_parameters=fit['active_parameters'],original_solver_unchanged=True)
 stagecost=[]
 for p in sorted((BASE/'execution'/family).glob('*/WORKER_EXIT.json')):
  x=read(p);assert x['exit_code']==0;stagecost.append(dict(stage=p.parent.name,wall_seconds=x['wall_seconds']))
 profile=read(ROOT/'artifacts/stvg_pipeline_speed_v1/student_profiles'/('followups64_'+family+'.json'));save('SOURCE_REUSE_COST.json',dict(actual_new_source_captures=profile['actual_new_source_captures'],source_prefix_restores=profile['source_prefix_restores'],actual_final_full_prefix_decoder_readouts=sum(x['actual_fit_count'] for x in noops),records=profile['rows'],GT_read=False));save('ACTUAL_CONFIGURATION.json',configs);save('NOOP_AND_USAGE.json',noops);save('COST.json',dict(stages=stagecost,fit_usage=read(b/'ROOT_STAGE_DIAGNOSTICS.json')['usage'],new_Teacher_calls=0,new_Temporal_fits=0,extra_qualification_fits=0,peak_GPU_memory_GiB=None,fresh_end_to_end_speedup=None,historical_reused_work_not_zero=True,completion_dense_reward_can_cost_more=family=='tube'))
 save('ROOT_SCALAR_READBACK.json',dict(status='pass',count=64,independent_checks=checks,all_signed_unknown_and_paired_t95_verified=True,all_final_tIoU_identical=True,no_new_model_or_GT_reader=True,GT_read=False))
 lines=['# '+('Fixed-budget coarse-to-fine OPD' if family=='anneal' else 'Sparse Teacher reference to complete Tube to Student: P0 feasibility'),'','HC32/VidSTG32 already exposed development parents, deterministic parent-hash selection. Homogeneous caption-fixed Native-clock B sparse Teacher references, identical existing T4 final intervals. No new Teacher or Temporal fit. No diffusion training, new confirmation, CURRENT promotion or automatic expansion.','', '| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@0.3 % | R@0.5 % |','|---|---|---:|---:|---:|---:|---:|']
 def val(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
 if family=='anneal':lines[2]+=' Primary matched contrasts are small and all corresponding vIoU source-paired t95 intervals include zero; this experiment does not establish an advantage for descending order. Historical HC original OPD also has two extra possible steps, so its higher mean is not an isolated noise-order result.'
 for ds in ('hc2','vidstg'):
  for a in arms:lines.append('|'+ds+'|'+a+'|'+'|'.join('unknown' if (v:=val(ds,a,k)) is None else f'{100*v:.6f}' for k in ('m_vIoU','m_sIoU_dense_GT','m_tIoU','R_vIoU_at_0p3','R_vIoU_at_0p5'))+'|')
 lines+=['','| Domain | Contrast | Metric | Delta pp | Source-paired t95 pp | + / - / neutral / unknown |','|---|---|---|---:|---|---|']
 for c in d['paired']:
  if c['metric'] in ('m_vIoU','m_sIoU_dense_GT','m_sIoU_anchor_GT','m_sIoU_native_nonanchor_GT','m_sIoU_interpolated_GT'):
   lines.append(f"|{c['dataset']}|{c['comparison']}|{c['metric']}|{c['mean_delta_pp'] if c['mean_delta_pp'] is not None else 'unknown'}|{c['paired_source_t95_interval_pp']}|{c['positive']}/{c['negative']}/{c['neutral']}/{c['unknown']}|")
 if family=='anneal':lines+=['','Primary comparison is descending against ascending and constant, all with three attempts and identical reject-rollback-continue semantics. Descending and ascending share the same three scales and standard-noise tensors per scale. Constant matches mean squared sigma. Original HC5/Vid3 early-stop OPD is historical only; differences from it combine retry/step/scale changes. Every stage recenters on actual Student output, and stage diagnostics report actual logit-anchor motion, protected-grid box motion and high-initial-pseudoreference anchor degradation. Smaller noise does not guarantee smaller actual movement or true-object correctness. No accepted bad step is allowed through the original sparse reference and protection gates.']
 else:lines+=['','P0_R is an external deterministic regularized residual proposal, not parameter TTA. P0_R_to_S is a genuine 1792-parameter Student fit through original Gaussian OPD/Adam/M2 with fixed sigma and original early stop. Original sparse Teacher acceptance and full-grid-minus-original-sparse protection are retained even though dense reward targets cover the full native grid. Missing references or invalid logit chart produce exact Source whole-query no-ops, retained in the denominator. Leave-one-reference checks use pseudoreferences and are not GT evidence. Dense reward/solver costs differ from sparse OPD; no equal-budget speed claim.']
 lines+=['','All64 predictions and full-prefix decoder readouts are sealed and independently audited before GT scoring. GT supports are identical between arms; anchor, other native-grid and interpolated known-GT strata retain unknowns. Metrics do not establish semantic identity or human review. Pairwise t95 intervals are descriptive development uncertainty, not a new selection confirmation. Source/feature reuse costs and actual new stage/solver costs are separated; peak memory/fresh end-to-end speedup remain unknown. No private media, GT coordinates, query/Caption text, raw Teacher/tensors/weights, credentials or personal conversations are exported.','', 'Actual settings are in ACTUAL_CONFIGURATION.json; fixed design and execution limitations are in the accompanying protocol.']
 doc=ROOT/'docs'/('STVG_OPD_ANNEAL64_001.md' if family=='anneal' else 'STVG_TUBE_COMPLETION64_P0_001.md');doc.write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(11,4));show=['constant','coarse_to_fine','reverse'] if family=='anneal' else ['P0_O','P0_R','P0_R_to_S']
 for ax,ds in zip(axs,('hc2','vidstg')):
  deltas=[100*(val(ds,a,'m_vIoU')-val(ds,'P0_F','m_vIoU')) for a in show];ax.bar(show,deltas);ax.axhline(0,color='black',lw=.8);ax.set(title=ds,ylabel='vIoU change over same-time Source (pp)');ax.tick_params(axis='x',rotation=18)
 fig.tight_layout();fig.savefig(out/'effects.png',dpi=160);plt.close(fig)
 for name in ('stvg_opd_followups64_v1.py','run_stvg_opd_followups64_v1.py','score_stvg_opd_followups64_v1.py','continue_stvg_opd_followups64_v1.py','continue_stvg_opd_followups64_002.py','report_stvg_opd_followups64_v1.py','publish_stvg_opd_followups64_v1.py','render_stvg_opd_followups64_v1.py','audit_stvg_tube_completion64_v1.py'):
  p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
 paths={str(doc.relative_to(ROOT)),'protocols/stvg_opd_anneal_and_tube64_v1.md'}|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()};verify_privacy(paths)
 for rel in paths:
  if (ROOT/rel).suffix=='.json':
   raw=(ROOT/rel).read_text()
   for cid in aliases:assert ('"'+cid+'"') not in raw,rel
 put(BASE/'publication'/family/'EXPORT_MANIFEST.json',dict(status='complete',count=64,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print(family.upper()+'_ANONYMOUS_EXPORT_PENDING_ACTUAL_ROOT_VIEW_AND_REMOTE')
if __name__=='__main__':main(sys.argv[1])
