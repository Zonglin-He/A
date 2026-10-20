"""Finite global initialization-strength calibration; offline, no model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import ROOT,D,P,BASE,read,put,sha,load,verify

def main(action):
 p=verify();G=[0.,.25,.5,.75,1.,1.25,1.5];stage=D/'initialization_strength'
 if action=='seal':
  assert not(stage/'PLAN_LOCK.json').exists();plan=dict(status='locked_before_strength_GT',gammas=G,expert_count=1,Source_Strong_and_original_consensus_fixed=True,meaning='scale the inherited consensus correction in center/log-size coordinates before OPD; gamma0 Source; gamma1 exact E_Ref',selection='one whole-panel mean-vIoU maximizer per dataset; ties closest gamma1',subsequent_action='if nonunit strength wins, verify it with actual matched frozen-decoder OPD before accepting a complete method',user_scope='latest authorization: root autonomously prioritizes and tunes parameters of both datasets',GPU_or_model_calls=0,epoch=time.time());put(stage/'PLAN_LOCK.json',plan);files={}
  for rec in p['records']:
   ds,a=rec['dataset'],rec['alias'];old=load(BASE/f'predictions/{ds}/{a}.pt');S=np.asarray(old['Source']);E=np.asarray(old['E_Ref']);out=dict(frame_ids=np.array(old['frame_ids']),positions=np.array(old['planned_positions']),Strong_interval=np.array(old['intervals']['Strong']))
   for gamma in G:
    b=S.copy()if gamma==0 else E.copy()if gamma==1 else np.concatenate((S[:,:2]+gamma*(E[:,:2]-S[:,:2]),S[:,2:]*np.exp(gamma*np.log(E[:,2:]/S[:,2:]))),-1);assert np.isfinite(b).all()and(b[:,2:]>0).all();out['G'+str(int(gamma*100)).zfill(3)]=b
   f=stage/'predictions'/ds/f'{a}.npz';f.parent.mkdir(parents=True,exist_ok=True);np.savez(f,**out);files[str(f.relative_to(ROOT))]=sha(f)
  put(stage/'PREDICTION_BARRIER.json',dict(status='sealed_all256_before_strength_GT',files=files,parents=256,epoch=time.time()));put(P/'INITIALIZATION_STRENGTH_PROTOCOL_PUBLIC.json',plan);print('INIT_GAMMA_SEAL',len(files),flush=True)
 else:
  import torch
  torch.set_num_threads(2)
  from scripts.score_stvg_opd_p1_v1 import truths
  from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
  from vg_tta.tastvg_paper48_metrics_v1 import xyxy
  bar=read(stage/'PREDICTION_BARRIER.json')
  for f,h in bar['files'].items():assert sha(ROOT/f)==h
  oldr=read(ROOT/'results/stvg_error_state/2026-10-20/single_referdino_bidirectional256/RESULTS_PUBLIC.json');out={};checks=0;errmax=0.
  for ds in ['hc2','vidstg']:
   records=[x for x in p['records']if x['dataset']==ds];raw=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'];gt,spans,prov=truths(ds,dict(parents=[x['query_ordinal']for x in records]));old={x['alias']:x for x in oldr['directions'][ds]['rows']};rows=[]
   for rec in records:
    a,q=rec['alias'],rec['query_ordinal'];r=raw[q];values={}
    with np.load(stage/'predictions'/ds/f'{a}.npz')as z:
     ids=z['frame_ids'].tolist();iv=z['Strong_interval'].tolist();pos=z['positions'].tolist();lookup={f:i for i,f in enumerate(ids)};fixed=[f for f in ids if f in gt[q]];unobs=[f for f in fixed if f not in {ids[i]for i in pos}]
     for gamma in G:
      arm='G'+str(int(gamma*100)).zfill(3);b=z[arm];tube=DenseTube(b,r,gt[q],spans[q],clip=ds=='hc2');v=tube.score(iv);off=official(b,r,gt[q],spans[q],iv,ds);e=max(abs(v[k]-off[k])for k in ['v','t','s']);assert e<2e-10;checks+=1;errmax=max(errmax,e);pix=xyxy(b,r['input']['width'],r['input']['height']);pix=np.maximum(pix,0)if ds=='hc2'else pix;node={f:float(box_iou(pix[lookup[f]],gt[q][f]))for f in fixed};v['fixed_actual_sIoU']=float(np.mean(list(node.values())));v['unobserved_actual_sIoU']=float(np.mean([node[f]for f in unobs]));values[arm]=v
      if gamma in [0,1]:
       reference=old[a]['values']['Temporal_only'if gamma==0 else'Without_spatial_adaptation'];assert all(abs(v[k]-reference[k])<2e-10 for k in v)
    rows.append(dict(alias=a,values=values))
   means={arm:{k:float(np.mean([x['values'][arm][k]for x in rows]))for k in rows[0]['values'][arm]}for arm in rows[0]['values']};ranked=sorted(means,key=lambda a:(-means[a]['v'],abs(int(a[1:])/100-1)));out[ds]=dict(parents=128,rows=rows,means=means,ranking=ranked,best_gamma=int(ranked[0][1:])/100,GT_provenance=prov,independent_confirmation=False)
  put(P/'INITIALIZATION_STRENGTH_RESULTS_PUBLIC.json',dict(status='completed_all256_global_strength_screen',datasets=out,official_dense_checks=checks,max_abs=errmax,new_model_calls=0,new_expert_calls=0,not_complete_adapted_method_results=True));print('INIT_GAMMA_SCORE', {ds:(q['best_gamma'],q['means'][q['ranking'][0]]['v'])for ds,q in out.items()},flush=True)
if __name__=='__main__':main(sys.argv[1])
