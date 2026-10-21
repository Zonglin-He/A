"""CPU-only complete-stage2 evaluation. A diagnostic pass is not a task result."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.evaluate import read,sha,write_json,score_boxes,mean_table,paired_stats,cost_summary
from scripts.propagation_v0.run import validate_phase,TRIAL
NAMES=['ERefStrong','GeometryOPD','EvidenceDP','MemoryGreedy','NoMemoryGreedy','OPD','FixedKD','Direct']
ARMS=['OPD','FixedKD','Direct']


def main():
 trial=TRIAL;lock=read(trial/'lock.json');receipts=validate_phase(trial,'stage2',lock)
 gate=read(trial/'gate.json');assert gate['passed']is True and gate['fingerprint']==lock['fingerprint']
 seal=read(trial/'stage2/sealed.json');before=sha(trial/'stage2/sealed.json')
 lp=ROOT/'data/evaluation/development_labels.jsonl';lh=sha(lp)
 assert lh==read(ROOT/'outputs/layout_reward_dev/scope_hc2/metrics.json')['labels_sha256']
 records=[json.loads(s)for s in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if s.strip()];records=[r for r in records if r['dataset']=='hc2'][:64]
 labels={r['id']:r for r in [json.loads(s)for s in lp.read_text().splitlines()if s.strip()]if r['id']in lock['ids']}
 rows=[];checks=0;err=0.;distance_rows=[]
 from scripts.propagation_v0.diagnose import distance_bins,bootstrap_mean
 from stvg.metrics import box_iou
 from stvg.metrics_base import xyxy
 for rec in records:
  key=rec['id'];g=labels[key];truth={int(k):v for k,v in g['boxes'].items()}
  with np.load(trial/'stage2'/seal['files'][key]['path'])as p:
   values={};node_values={};positions=p['positions8'].tolist()
   for name in NAMES:
    values[name],e,support=score_boxes(p[name],rec,truth,p['Strong_interval'].tolist(),g['interval'],p['positions8'].tolist());checks+=1;err=max(err,e)
    pixels=np.maximum(xyxy(p[name],rec['metadata']['width'],rec['metadata']['height']),0)
    node_values[name]=[None if int(fid)not in truth else box_iou(pixels[t],truth[int(fid)])for t,fid in enumerate(p['frame_ids'])]
  bins=distance_bins(rec['metadata']['frame_ids'],positions)
  dr={}
  for group in ('near','middle','far'):
   idx=[int(t)for t in np.flatnonzero(bins[group])if node_values['ERefStrong'][int(t)]is not None]
   dr[group]={a:None if not idx else float(np.mean([node_values[a][t]-node_values['ERefStrong'][t]for t in idx]))for a in ARMS}
  distance_rows.append(dr)
  rows.append(dict(id=key,parent=g['parent'],values=values,support=support))
 assert len({r['parent']for r in rows})==len(rows)==64 and sha(lp)==lh
 validate_phase(trial,'stage2',lock);assert sha(trial/'stage2/sealed.json')==before
 means,counts=mean_table(rows,NAMES);signed=[]
 for i,r in enumerate(rows):
  for arm in ARMS:
   for base in ['ERefStrong','GeometryOPD','EvidenceDP','FixedKD','Direct']:
    if arm==base:continue
    a,b=r['values'][arm],r['values'][base]
    signed.append(dict(query=f'H{i+1:03d}',method=arm,baseline=base,vIoU_delta_pp=100*(a['v']-b['v']),unobserved_sIoU_delta_pp=None if a['unobserved_actual_sIoU']is None or b['unobserved_actual_sIoU']is None else 100*(a['unobserved_actual_sIoU']-b['unobserved_actual_sIoU']),severe_decline=bool(a['v']-b['v']<-.05)))
 proof=dict(fingerprint=lock['fingerprint'],stage2_seal_sha256=before,gate_sha256=sha(trial/'gate.json'),development_gate_used=True,labels_sha256=lh,complete_predictions_validated_before_labels=True,official_dense_checks=checks,official_dense_max_absolute_error=err,GT_used_for_query_prediction=False)
 public=dict(status='completed_HC64_v0_exposed_development',queries=64,parents=64,config=lock['config'],means=means,valid_query_counts=counts,paired=paired_stats(rows,NAMES),costs=cost_summary(receipts,ARMS),proof=proof,queries_anonymous=[dict(query=f'H{i+1:03d}',values=r['values'])for i,r in enumerate(rows)],signed_changes=signed,distance={g:{a:bootstrap_mean([r[g][a]for r in distance_rows])for a in ARMS}for g in ('near','middle','far')})
 write_json(trial/'stage2/metrics.json',{**public,'rows':rows})
 dest=ROOT/'results/propagation_v0_hc64';dest.mkdir(parents=True,exist_ok=True);write_json(dest/'adaptation.json',public)
 lines=['# v0 adapted propagation: HC64','','These are complete prelocked development results after the whole-panel diagnostic gate. The gate used development GT; it never selected individual predictions or parameters.','', '| Method | vIoU % | Observed sIoU % | Unobserved sIoU % |','|---|---:|---:|---:|']
 for name in NAMES:
  m=means[name];fmt=lambda a:'NA'if a is None else f'{100*a:.4f}'
  lines.append('| '+name+' | '+' | '.join(fmt(m[k])for k in ['m_vIoU','m_sIoU_observed_actual','m_sIoU_unobserved_actual'])+' |')
 lines += ['','Complete signed outcomes, paired parent confidence intervals, accepted/rejected steps and actual cached incremental costs are in adaptation.json. Fixed-score rollback does not prove true localization improvement. Continuous Direct uses the same actual hard final decode, not averaged output boxes.']
 (dest/'ADAPTATION.md').write_text('\n'.join(lines)+'\n')
 write_json(trial/'stage2/score_receipt.json',dict(status='scored',proof=proof,metrics_sha256=sha(trial/'stage2/metrics.json'),public_sha256=sha(dest/'adaptation.json')))
 print(json.dumps(dict(status='scored',means=means)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-after-seal',action='store_true');a=p.parse_args();assert a.run_after_seal;main()
