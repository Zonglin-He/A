"""Anonymous paper integration / independent QCQP group exports and scalar audit."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.stvg_paper_public_privacy_v1 import verify_privacy
BASES={'paper':ROOT/'artifacts/stvg_paper_t2_b_v1','qcqp':ROOT/'artifacts/stvg_output_budget32_v1'}
CODES=['stvg_paper_t2_b_v1.py','run_stvg_paper_t2_b_v1.py','score_stvg_paper_t2_b_v1.py','stvg_output_budget_qp_v1.py','run_stvg_output_budget32_v1.py','score_stvg_output_budget32_v1.py','continue_stvg_output_budget32_v1.py','audit_stvg_b_score_algebra_v1.py','report_stvg_paper_frozen_v1.py','publish_stvg_paper_frozen_v1.py','stvg_paper_public_privacy_v1.py']
def main(kind):
 import numpy as np
 from scipy.stats import t
 base=BASES[kind];n=256 if kind=='paper' else 32;domain_n=n//2;done=read(base/'CPU_COMPLETION.json');assert done['status']=='complete' and done['count']==n and done['result_sha256']==sha(base/'POSTSEAL_RESULTS.json');d=read(base/'POSTSEAL_RESULTS.json');rr=read(base/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)};idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};arms=list(dict.fromkeys(r['arm'] for r in d['rows']));assert len(idx)==len(d['rows'])==n*len(arms);checks=0
 for s in d['summary']:
  ids=[r['probe_alias'] for r in rr if r['dataset']==s['dataset']];vals=[idx[i,s['arm']][s['metric']] for i in ids if idx[i,s['arm']][s['metric']] is not None];assert len(vals)==s['valid'] and domain_n-len(vals)==s['unknown'];assert (not vals and s['mean'] is None) or abs(math.fsum(vals)/len(vals)-s['mean'])<1e-12;checks+=1
 for c in d['paired']:
  a,b=c['comparison'].split('_minus_');ids=[r['probe_alias'] for r in rr if r['dataset']==c['dataset']];v={i:100*(idx[i,a][c['metric']]-idx[i,b][c['metric']]) for i in ids if idx[i,a][c['metric']] is not None and idx[i,b][c['metric']] is not None};assert v==c['per_source_delta_pp'] and len(v)==c['valid'] and domain_n-len(v)==c['unknown'];mean=math.fsum(v.values())/len(v) if v else None;assert mean==c['mean_delta_pp']
  if len(v)>1:
   err=float(t.ppf(.975,len(v)-1)*np.std(list(v.values()),ddof=1)/math.sqrt(len(v)));assert np.allclose([mean-err,mean+err],c['paired_source_t95_interval_pp'],atol=1e-11,rtol=0)
  assert (c['positive'],c['negative'],c['neutral'])==(sum(x>1e-6 for x in v.values()),sum(x< -1e-6 for x in v.values()),sum(abs(x)<=1e-6 for x in v.values()));checks+=len(v)
 def anon(x):
  if isinstance(x,list):return [anon(z) for z in x]
  if isinstance(x,dict):return {aliases.get(k,k):anon(v) for k,v in x.items() if k!='GT_bindings'}
  return aliases.get(x,x) if isinstance(x,str) else x
 out=ROOT/'results/stvg_frozen_paper/2026-10-17'/kind;out.mkdir(parents=True,exist_ok=False)
 def save(name,x):write(out/name,anon(x))
 save('RESULTS.json',d);save('METHOD_LOCK.json',read(base/'METHOD_LOCK.json'));save('ROOT_SCALAR_READBACK.json',dict(status='pass',count=n,checks=checks,complete_signed_unknown_and_paired_intervals=True,no_new_model_or_GT_reader=True,GT_read=False))
 stages=[]
 for p in sorted((base/'execution').glob('*/WORKER_EXIT.json')):
  v=read(p);assert v['exit_code']==0;stages.append(dict(stage=p.parent.name,wall_seconds=v['wall_seconds']))
 cost=dict(actual_stages=stages,new_Temporal_fit_calls=0,new_Caption_calls=0,previously_reused_work_is_not_zero=True,fresh_matched_end_to_end_speedup=None,not_new_confirmation=True)
 if kind=='paper':
  feature=read(base/'FEATURE_BARRIER.json');teacher=read(base/'arms/B/teacher/G32/TEACHER_GLOBAL_BARRIER.json');qual=read(base/'arms/B/NATIVE_QUALIFICATION.json');cost.update(new_spatial_WHO_calls=teacher['new_generation_calls'],extra_qualification_generations=qual['additional_generation_calls'],new_visual_encoder_calls=feature['new_encoder_calls'],exact_visual_reuses=feature['exact_visual_reuses']);root=read(base/'arms/B/trials/G32/ROOT_READBACK.json');save('ROOT_READBACK.json',root);save('USAGE.json',d['usage'])
 else:
  save('ROOT_READBACK.json',read(base/'ROOT_READBACK.json'));save('QUALIFICATION_COMPLETION.json',read(base/'QUALIFICATION_COMPLETION.json'));save('USAGE.json',read(base/'ROOT_USAGE.json'));cost.update(new_spatial_WHO_calls=0,dense_Jacobian_and_dual_cost_in_fit_usage=True,peak_GPU_memory_GiB=None)
 save('COST.json',cost)
 def val(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
 lines=['# '+('Frozen paper T2+B integrated256' if kind=='paper' else 'Independent output-KL-budget M0/M1/M2 development32'),'','This is a frozen development evaluation with all no-fit/unknown outcomes retained. Target GT was read only after complete prediction/root seals. It is not independent confirmation. Historical failed confirmation and T2/T4 noninferiority findings remain unchanged.','', '| Domain | Arm | vIoU % | tIoU % | dense sIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|']
 for ds in ('hc2','vidstg'):
  for a in arms:lines.append('|'+ds+'|'+a+'|'+'|'.join('unknown' if (v:=val(ds,a,k)) is None else f'{100*v:.6f}' for k in ('m_vIoU','m_tIoU','m_sIoU_dense_GT','R_vIoU_at_0p3','R_vIoU_at_0p5'))+'|')
 lines+=['','| Domain | Contrast | Metric | Delta pp | Source-paired t95 pp | + / - / neutral / unknown |','|---|---|---|---:|---|---|']
 for c in d['paired']:
  if c['metric'] in ('m_vIoU','m_tIoU','m_sIoU_dense_GT'):lines.append(f"|{c['dataset']}|{c['comparison']}|{c['metric']}|{c['mean_delta_pp']}|{c['paired_source_t95_interval_pp']}|{c['positive']}/{c['negative']}/{c['neutral']}/{c['unknown']}|")
 if kind=='paper':lines+=['','Full_T2_B uses actual T2-conditioned observations, candidate pixels, Caption policy and WHO references, followed by original sparse Gaussian OPD. Historical T4 Full is explicitly labeled; NativeB evidence attached to final T2 time is a readout diagnostic. T2 was selected by human simplicity preference, not a relabeled noninferiority claim. B is fixed, with s=.75mean-.125disagreement and original MMR scale; it is not a calibrated posterior or universal superiority claim. No C/Global32/noCoverage/annealing/Tube/Decoder-Aligned or online writeback. Source/T2/cached feature work is reused only after exact dependency checks and is not fresh zero-cost work. Official full roster, scientifically qualified second Backbone and fresh cost comparison are separate actual execution tasks, not completed by this256 result.']
 else:lines+=['','M0 is the exact old original-response OPD. M1 projects raw Adam onto the fixed linearized output-KL budget; M2 realizes the OPD anchor target with the same budget and gamma=1. Epsilon=.01 is fixed without GT. All accepted M1/M2 steps also satisfy actual nonlinear decoded KL and original sparse reference acceptance, with original rollback/early-stop and no added shrink search. Invalid whole-trajectory logit charts are whole-query Source no-ops in the full32 table; the pre-adaptation chart-valid diagnostic never replaces it. Primary contrast is M2-M1. This independent branch does not change the frozen paper method. Dense-Jacobian implementation costs and unknown fresh/peak comparisons are explicit. No new Teacher/Temporal fits or policy/Prompt/LR/sigma search.']
 lines+=['','All source-paired signs and unknowns are public. Pixels are separately root inspected; these engineering views are not independent semantic labels. Private media/query/Caption text, GT coordinates, raw Teacher/tensors/weights, credentials and personal conversations are excluded.']
 doc=ROOT/'docs'/('STVG_PAPER_T2_B256_001.md' if kind=='paper' else 'STVG_OUTPUT_BUDGET32_001.md');doc.write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 show=['T2_only','S_B_Native','Full_T2_B'] if kind=='paper' else ['M0','M1','M2'];ref='Frozen' if kind=='paper' else 'Source_T';fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,ds in zip(axs,('hc2','vidstg')):ax.bar(show,[100*(val(ds,a,'m_vIoU')-val(ds,ref,'m_vIoU')) for a in show]);ax.axhline(0,color='black',lw=.8);ax.set(title=ds,ylabel='vIoU change (pp)');ax.tick_params(axis='x',rotation=15)
 fig.tight_layout();fig.savefig(out/'effects.png',dpi=160);plt.close(fig)
 for name in CODES:
  p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
 extra={'protocols/stvg_paper_t2_b_v1.md','protocols/stvg_output_budget32_v1.md','methods/PAPER_METHOD.json','methods/paper/STVG_T2_B_EPISODIC_V1.json','paper/README_CURRENT.md','paper/stvg_t2_b_episodic_20261017.tex','docs/STVG_B_SCORE_ALGEBRA_CORRECTION_20261017.md'}
 if kind=='paper':
  p=ROOT/'paper/stvg_t2_b_episodic_20261017.tex';s=p.read_text();s=s.replace('Full T2+B &pending&pending',f"Full T2+B &{100*val('hc2','Full_T2_B','m_vIoU'):.4f}&{100*val('vidstg','Full_T2_B','m_vIoU'):.4f}");s=s.replace('The integrated two-candidate Full evaluation is being executed under a frozen manifest; the manuscript does not substitute historical four-candidate results for that evaluation.','The integrated two-candidate Full evaluation uses actual newly executed spatial evidence and complete Student outputs; historical four-candidate results remain separate controls.');p.write_text(s)
 paths=extra|{str(doc.relative_to(ROOT))}|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()};verify_privacy(paths);write(base/'publication'/kind/'EXPORT_MANIFEST.json',dict(status='complete',count=n,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print(kind.upper()+'_ANONYMOUS_EXPORT_PENDING_ROOT_PIXELS_REMOTE')
if __name__=='__main__':main(sys.argv[1])
