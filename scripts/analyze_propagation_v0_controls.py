"""Audit saved v0 control scores against saved task metrics; no new predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import read,sha,atomic
from scripts.propagation_v0.run import validate_phase


def main():
 tr=ROOT/'outputs/propagation_v0_hc64';lock=read(tr/'lock.json');receipts=validate_phase(tr,'stage1',lock)
 public=read(ROOT/'results/propagation_v0_hc64/diagnostics.json');assert public['proof']['stage1_seal_sha256']==sha(tr/'stage1/sealed.json')
 names=['MemoryGreedy','NoMemoryGreedy','EvidenceDP']
 stats={n:dict(score_increased=0,score_increased_but_vIoU_declined=0,total_score_gain=0.,vIoU_declines=0,severe_vIoU_declines=0)for n in names};times={};rows=[]
 for i,key in enumerate(lock['ids']):
  rr=receipts[key];trace=torch.load(tr/'stage1'/rr['trace_path'],weights_only=False,map_location='cpu');bank=torch.load(tr/'stage1'/rr['bank_path'],weights_only=False,map_location='cpu')
  base=float(bank['score_unary'][:,0].sum()+bank['score_pair'][:,0,0].sum());r={}
  for n in names:
   path=trace['controls'][n];a=path['actions']if 'actions'in path else path['trace']['actions'];score=float(bank['score_unary'][torch.arange(len(a)),a].sum()+bank['score_pair'][torch.arange(len(a)-1),a[:-1],a[1:]].sum())
   dv=100*(public['queries_anonymous'][i]['values'][n]['v']-public['queries_anonymous'][i]['values']['ERefStrong']['v']);ds=stats[n]
   ds['score_increased']+=int(score>base+1e-12);ds['score_increased_but_vIoU_declined']+=int(score>base+1e-12 and dv < -1e-8);ds['total_score_gain']+=score-base;ds['vIoU_declines']+=int(dv< -1e-8);ds['severe_vIoU_declines']+=int(dv< -5);r[n]=dict(score_delta=score-base,vIoU_delta_pp=dv)
   if n=='EvidenceDP':assert score>=base-1e-12
  for n,v in {**rr['timings'],'total_seconds':rr['total_seconds']}.items():times.setdefault(n,[]).append(v)
  rows.append(dict(query=f'H{i+1:03d}',values=r))
 validate_phase(tr,'stage1',lock)
 summary=dict(status='sealed_controls_offline_readback_no_new_predictions',queries=64,adaptation_executed=False,
  gate=read(tr/'gate.json')['decision'],proof=dict(stage1_seal_sha256=sha(tr/'stage1/sealed.json'),new_label_reads=0,uses_saved_GT_metrics=True,new_rollouts=0,new_model_calls=0),
  fixed_score_vs_GT=stats,costs={k:dict(mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9)),total=float(np.sum(v)))for k,v in times.items()},
  stage1_run_seconds=read(tr/'stage1/run.json')['seconds'],queries_anonymous=rows)
 atomic(ROOT/'results/propagation_v0_hc64/control_audit.json',summary)
 print(json.dumps({k:v for k,v in summary.items()if k!='queries_anonymous'}))
if __name__=='__main__':main()
