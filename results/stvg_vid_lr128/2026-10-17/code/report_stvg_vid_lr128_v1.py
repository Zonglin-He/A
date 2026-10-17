"""Complete anonymous Vid LR comparison and independent scalar audit."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_vid_lr128_v1 import BASE,read,sha,put
def main():
 import numpy as np
 from scipy.stats import t
 from scripts.export_stvg_lite_batch001 import public_source_text
 from scripts.publish_stvg_soft_ep_v1 import verify_privacy
 c=read(BASE/'CPU_COMPLETION.json');assert c['status']=='complete' and c['count']==128 and c['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
 d=read(BASE/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)};ix={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};arms=list(dict.fromkeys(r['arm'] for r in d['rows']));assert len(ix)==128*len(arms)==len(d['rows']);checks=0
 for s in d['summary']:
  vs=[ix[c,s['arm']][s['metric']] for c in aliases if ix[c,s['arm']][s['metric']] is not None];assert len(vs)==s['valid'] and 128-len(vs)==s['unknown'];assert (not vs and s['mean'] is None) or abs(math.fsum(vs)/len(vs)-s['mean'])<1e-12;checks+=1
 for p in d['paired']:
  a,b=p['comparison'].split('_minus_');v={c:100*(ix[c,a][p['metric']]-ix[c,b][p['metric']]) for c in aliases if ix[c,a][p['metric']] is not None and ix[c,b][p['metric']] is not None};assert v==p['per_source_delta_pp'];assert len(v)==p['valid'] and 128-len(v)==p['unknown'];mean=math.fsum(v.values())/len(v);assert mean==p['mean_delta_pp'];e=float(t.ppf(.975,len(v)-1)*np.std(list(v.values()),ddof=1)/math.sqrt(len(v)));assert np.allclose([mean-e,mean+e],p['paired_source_t95_interval_pp'],rtol=0,atol=1e-11);assert (sum(x>1e-6 for x in v.values()),sum(x< -1e-6 for x in v.values()),sum(abs(x)<=1e-6 for x in v.values()))==(p['positive'],p['negative'],p['neutral']);checks+=len(v)
 def anon(v):
  if isinstance(v,dict):return {aliases.get(k,k):anon(x) for k,x in v.items() if k!='GT_bindings'}
  if isinstance(v,list):return [anon(x) for x in v]
  return aliases.get(v,v) if isinstance(v,str) else v
 out=ROOT/'results/stvg_vid_lr128/2026-10-17';out.mkdir(parents=True,exist_ok=False)
 for name,x in [('RESULTS.json',d),('METHOD_LOCK.json',read(BASE/'METHOD_LOCK.json')),('MATCHED_QUALIFICATION_COMPLETION.json',read(BASE/'MATCHED_QUALIFICATION_COMPLETION.json')),('ROOT_SCALAR_READBACK.json',dict(status='pass',count=128,checks=checks,all_signed_unknown_paired_intervals_preserved=True,new_GT_reader=False))]:put(out/name,anon(x))
 for arm in ('LR_HALF','LR_QUARTER'):
  b=BASE/'arms'/arm/'trials/G32';design=read(b/'DESIGN_LOCK.json')
  put(out/(arm+'_CONFIG.json'),dict(config=design['configs']['vidstg']['M2'],response_settings=design['response_settings']['vidstg']['M2'],recipe='lite_fullstep',episodic_parameters=1792,GT_read=False))
  put(out/(arm+'_ROOT_READBACK.json'),read(b/'ROOT_READBACK.json'))
 costs=[dict(arm=p.parent.parent.name,stage=p.parent.name,**read(p)) for p in (BASE/'execution').glob('*/*/WORKER_EXIT.json')];assert all(x['exit_code']==0 for x in costs);put(out/'COST.json',dict(stages=costs,new_WHO=0,new_Temporal_fits=0,maximum_new_Spatial_fits=256,baseline='exact sealed .027 reuse; old costs are not zero',fresh_end_to_end_speedup=None))
 lines=['# Vid128 fixed-reference Spatial learning-rate development comparison','','Same exposed core256 Vid128 parents. T2 clock, Caption, WHO references, Gaussian32 samples and query/round seeds, sigma=.4, tau=.5, steps3, response mu=.125 and original fullstep acceptance/rollback remain fixed. Current LR=.027 is exact whole-query sealed reuse; only .0135/.00675 are newly fit. No new Teacher/Temporal. GT read only after both complete prediction/root seals. This is development tuning, not new confirmation; all128 including no-fit are retained.','','| Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |','|---|---:|---:|---:|---:|---:|']
 def val(a,k):return next(s['mean'] for s in d['summary'] if s['arm']==a and s['metric']==k)
 for a in arms:lines.append('|'+a+'|'+'|'.join('unknown' if (v:=val(a,k)) is None else f'{100*v:.6f}' for k in ['m_vIoU','m_sIoU_dense_GT','m_tIoU','R_vIoU_at_0p3','R_vIoU_at_0p5'])+'|')
 lines+=['','| Contrast | vIoU delta pp | Paired source t95 pp | + / - / neutral / unknown |','|---|---:|---|---|']
 for p in d['paired']:
  if p['metric']=='m_vIoU':lines.append(f"|{p['comparison']}|{p['mean_delta_pp']}|{p['paired_source_t95_interval_pp']}|{p['positive']}/{p['negative']}/{p['neutral']}/{p['unknown']}|")
 lines+=['','Development selection (all candidates disclosed): '+json.dumps(d['development_selection']), '', 'Selection uses the already-exposed development set. Post-selection confidence intervals are descriptive, not unbiased confirmation of the chosen arm. No frozen paper or historical CURRENT registry is overwritten; no automatic follow-up search. Full updates may still follow imperfect Teacher references. Actual engineering pixel panels do not complete independent semantic review. Raw media, GT coordinates, actual query/Caption text, tensors and weights remain private.']
 doc=ROOT/'docs/STVG_VID_LR128_001.md';doc.write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 show=['LR_CURRENT','LR_HALF','LR_QUARTER'];fig,ax=plt.subplots(figsize=(7,4));ax.bar(show,[100*(val(a,'m_vIoU')-val('T2_only','m_vIoU')) for a in show]);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('vIoU change from T2-only (pp)');fig.tight_layout();fig.savefig(out/'effects.png',dpi=160);plt.close(fig)
 for name in ['stvg_vid_lr128_v1.py','run_stvg_vid_lr128_v1.py','report_stvg_vid_lr128_v1.py','publish_stvg_vid_lr128_v1.py','render_stvg_vid_lr128_v1.py']:
  p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
 paths={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_vid_lr128_v1.md'};verify_privacy(paths);put(BASE/'publication/lr128/EXPORT_MANIFEST.json',dict(status='complete',count=128,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('VID_LR128_EXPORT_READY')
if __name__=='__main__':main()
