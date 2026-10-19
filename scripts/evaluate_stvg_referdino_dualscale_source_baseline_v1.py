"""Cached no-spatial-adaptation Source with IDENTICAL fixed MBR time.

Never used to select policy. Lock is CPU/cache only before new evaluation;
score requires both the model-prediction and this baseline's sealed barriers.
"""
from pathlib import Path
import os,sys,time,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha

def main(stage,confirmation):
 if confirmation:
  from scripts.run_stvg_referdino_dualscale_confirm32_v1 import D,P,verify
  inputs=D/'inputs'
 else:
  from scripts.run_stvg_referdino_dualscale_hc128_v1 import D,P,PRE,verify
  inputs=PRE/'inputs'
 p=verify(True);n=len(p['records']);B=D/'source_baseline'
 if stage=='lock':
  files=[]
  for r in p['records']:
   z=load(inputs/(r['alias']+'.pt'));assert not z['GT_read'];files.append(save(B/(r['alias']+'.pt'),dict(alias=r['alias'],dataset='hc2',query_ordinal=r['query_ordinal'],frame_ids=z['frame_ids'],interval=z['interval'],planned_positions=r['positions8'],outputs=dict(E_MBR=z['E'],Source_MBR=z['Source'].double().numpy()),GT_read=False,optimizer_updates=0)))
  put(D/'SOURCE_BASELINE_BARRIER.json',dict(count=n,files=files,new_model_calls=0,GT_read=False,scope='no spatial adaptation; SAME fixed E finalMBR time, not original native-time output'));print('CACHED SOURCE LOCKED',n,flush=True);return
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 torch.set_num_threads(4);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==n;bar=read(D/'SOURCE_BASELINE_BARRIER.json');assert bar['count']==n
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 pub=read(P/'RESULTS_PUBLIC.json');rr={r['alias']:r for r in pub['rows']};selected=read(D/'PLAN_LOCK.json')['selected_policy']if confirmation else read(D/'SELECTION_LOCK.json')['selected_policy'];stem='Selected_Live'if confirmation else selected+'_Live';oldstem='Old_Live';gt,spans,_=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));plans=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rows=[];checks=0
 for r in p['records']:
  z=load(B/(r['alias']+'.pt'));q=r['query_ordinal'];m=metrics(z,plans[q],gt[q],spans[q]);assert m['values']['E_MBR']==rr[r['alias']]['values']['E_MBR'];source=m['values']['Source_MBR'];values=dict(Source_MBR=source,E_MBR=m['values']['E_MBR'],Direct=rr[r['alias']]['values']['Direct'],Selected_Live_seedmean=rr[r['alias']]['values'][stem+'_seedmean'],Old_Live_seedmean=rr[r['alias']]['values'][oldstem+'_seedmean'])
  for seed in p['seeds']:values[f'Selected_Live_{seed}']=rr[r['alias']]['values'][f'{stem}_{seed}']
  independently=DenseTube(z['outputs']['Source_MBR'],plans[q],gt[q],spans[q],clip=True).score(z['interval'])
  for ep in ['v','t','s']:assert abs(independently[ep]-source[ep])<2e-10;checks+=1
  rows.append(dict(alias=r['alias'],values=values))
 eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];pairs=['E_MBR','Direct','Old_Live_seedmean','Selected_Live_seedmean']+[f'Selected_Live_{s}'for s in p['seeds']];stats={}
 for arm in pairs:
  stats[arm+'_minus_Source_MBR']={}
  for ep in eps:
   data=[None if row['values'][arm][ep]is None or row['values']['Source_MBR'][ep]is None else 100*(row['values'][arm][ep]-row['values']['Source_MBR'][ep])for row in rows];v=paired(data);stats[arm+'_minus_Source_MBR'][ep]=v;xs=np.asarray([x for x in data if x is not None]);idx=np.random.default_rng(20261018).integers(len(xs),size=(10000,len(xs)));assert abs(float(xs.mean())-v['mean'])<1e-12 and np.allclose(np.quantile(xs[idx].mean(1),[.025,.975]),v['CI95'],atol=1e-12,rtol=0);checks+=1
 put(P/'SOURCE_BASELINE_PUBLIC.json',dict(status='pass',count=n,scope='Source spatial output, same frozen finalMBR temporal support',not_used_for_selection=True,selected_policy=selected,paired=stats,rows=rows,new_model_or_expert_calls=0,independent_DenseTube_and_parent_bootstrap_checks=checks,all_source_readouts_locked_before_new_scoring=True,no_claim_that_total_spatial_gain_equals_OPD_gain=True))
 lines=['# Complete spatial contribution relative to Source + same MBR','', 'This is no SPATIAL adaptation at identical fixed temporal support. It is not a claim about the original native-time Source output. Source coordinates come only from pinned cached Source predictions; no network call. This comparison is excluded from policy selection. E−Source and selectedLive−E must remain separate when assigning contribution.','', '| Comparison | Full vIoU pp [95%CI] | Outside8 actual sIoU pp [95%CI] |','|---|---:|---:|']
 for arm,ss in stats.items():
  def f(x):return f"{x['mean']:+.4f} [{x['CI95'][0]:+.4f},{x['CI95'][1]:+.4f}]"
  lines.append(f"| {arm} | {f(ss['v'])} | {f(ss['unobserved_actual_s'])} |")
 (P/'SOURCE_BASELINE_REPORT.md').write_text('\n'.join(lines)+'\n');assert not torch.cuda.is_initialized();print('SOURCE BASELINE SCORED',n,checks,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['lock','score']);a.add_argument('--confirmation',action='store_true');x=a.parse_args();main(x.stage,x.confirmation)
