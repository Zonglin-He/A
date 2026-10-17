"""Independent saved-scalar audit and anonymous geometry/config factorial export."""
from pathlib import Path
import math,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_geometry_config32_v1 import BASE,read,sha,put
GROUP='factorial32'
def main():
 import numpy as np
 from scipy.stats import t
 from scripts.export_stvg_lite_batch001 import public_source_text
 from scripts.publish_stvg_soft_ep_v1 import verify_privacy
 assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
 d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['case_id']:f'R{i:03d}' for i,r in enumerate(rr,1)};idx={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};checks=0
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
 for p in d['interactions']:
  pre=p['readout'];k=p['metric'];v={r['case_id']:100*((idx[r['case_id'],pre+'DINO_common_ORIGINAL'][k]-idx[r['case_id'],pre+'DINO_common_SHORT'][k])-(idx[r['case_id'],pre+'WF_common_ORIGINAL'][k]-idx[r['case_id'],pre+'WF_common_SHORT'][k])) for r in rr if r['dataset']==p['dataset']};audit(v,p)
 def anon(v):
  if isinstance(v,dict):return {aliases.get(k,k):anon(x) for k,x in v.items() if k!='GT_bindings'}
  if isinstance(v,list):return [anon(x) for x in v]
  return aliases.get(v,v) if isinstance(v,str) else v
 out=ROOT/'results/stvg_geometry_config32/2026-10-17';out.mkdir(parents=True,exist_ok=False)
 for name,obj in [('RESULTS.json',d),('METHOD_LOCK.json',read(BASE/'METHOD_LOCK.json')),('ROOT_READBACK.json',read(BASE/'ROOT_READBACK.json')),('QUALIFICATION_COMPLETION.json',read(BASE/'QUALIFICATION_COMPLETION.json')),('ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,full_signed_unknown_interaction_audit=True,new_GT_reader=False))]:put(out/name,anon(obj))
 costs=[dict(stage=p.parent.name,**read(p)) for p in (BASE/'execution').glob('*/WORKER_EXIT.json')];assert all(r['exit_code']==0 for r in costs);bar=read(BASE/'STUDENT_GLOBAL_BARRIER.json');put(out/'COST.json',dict(stages=costs,new_scientific_fits=bar['new_scientific_fits'],new_qualification_fits=bar['new_extra_qualification_fits'],Source_capture_calls=bar['source_capture_calls'],extra_RGB_qualification_calls=2,new_Teacher=0,new_DINO=0,short_cells='exact sealed historical reuse; historical cost is not zero',fresh_end_to_end_cost=None))
 lines=['# Same-support reference geometry × configuration32','','Historical exposed HC16/Vid16,38 common anchors in19 sources;13 unsupported no-op sources retained. Identical support/weights/Source/Adam reset/seeds/final intervals. Short cells are exact sealed reuse; original-package cells are newly executed. All episodic, no LN carry. Historical current5 is not current paper Lite. Original package changes lr/sigma/tau/steps together; not a steps-only ablation or an online-stream reproduction. All new predictions and independent math/output ROOT seals preceded GT scoring.','','| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |','|---|---|---:|---:|---:|---:|---:|']
 def val(ds,a,k):return next(s['mean'] for s in d['summary'] if s['dataset']==ds and s['arm']==a and s['metric']==k)
 arms=list(dict.fromkeys(r['arm'] for r in d['rows']))
 for ds in ['hc2','vidstg']:
  for a in arms:lines.append('|'+ds+'|'+a+'|'+'|'.join('unknown' if (v:=val(ds,a,k)) is None else f'{100*v:.6f}' for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
 lines+=['','Primary interaction: (DINO_original−DINO_short)−(WF_original−WF_short). Same final temporal evaluation support within each S or TS readout.','','| Domain/readout/metric | Interaction pp | Paired source t95 pp | + / − / neutral / unknown |','|---|---:|---|---|']
 for p in d['interactions']:lines.append(f"|{p['dataset']}/{p['readout']}/{p['metric']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
 lines+=['','| Domain | TS contrast | vIoU delta pp | Paired source t95 pp | + / − / neutral / unknown |','|---|---|---:|---|---|']
 for p in d['paired']:
  if p['metric']=='m_vIoU' and p['comparison'].startswith('TS_') and ('_minus_T' in p['comparison'] or '_minus_TS_' in p['comparison']):lines.append(f"|{p['dataset']}|{p['comparison']}|{p['mean_delta_pp']:.6f}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
 lines+=['','All pairwise source deltas, signs, unknowns and intervals for all metrics are in RESULTS.json; t95 is descriptive on this exposed small development sample. The support is conditional on historical DINO admission. This factorial tests geometry × configuration package, not an isolated step-count mechanism, stronger-MLLM claim, current-B superiority or long-stream online state. Sparse-reference acceptance is not GT correctness. Teacher-reference geometry quality was evaluated previously and is not rerun here. Actual pixel review is engineering inspection, not completion of the512-window semantic panel. Production CURRENT and frozen paper registry remain unchanged.']
 doc=ROOT/'docs/STVG_GEOMETRY_CONFIG32_001.md';doc.write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4));shown=['TS_WF_common_SHORT','TS_DINO_common_SHORT','TS_WF_common_ORIGINAL','TS_DINO_common_ORIGINAL']
 for ax,ds in zip(axs,['hc2','vidstg']):ax.bar(['WF short','DINO short','WF original','DINO original'],[100*(val(ds,a,'m_vIoU')-val(ds,'T','m_vIoU')) for a in shown]);ax.axhline(0,color='black',lw=.7);ax.set_title(ds+' (all16 sources)');ax.set_ylabel('vIoU delta from T (pp)');ax.tick_params(axis='x',rotation=25)
 fig.tight_layout();fig.savefig(out/'effects.png',dpi=160);plt.close(fig)
 for name in ['stvg_geometry_config32_v1.py','run_stvg_geometry_config32_v1.py','report_stvg_geometry_config32_v1.py','render_stvg_geometry_config32_v1.py','publish_stvg_geometry_config32_v1.py']:
  p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
 paths={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_geometry_config32_v1.md'};verify_privacy(paths);put(BASE/'publication'/GROUP/'EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('GEOMETRY_CONFIG32_EXPORT_READY',checks)
if __name__=='__main__':main()
