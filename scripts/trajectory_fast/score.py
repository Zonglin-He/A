"""Independent CPU scoring for sealed sparse/batched trajectory-evidence HC64 experiment with256 adaptation evaluation allowance."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.evaluate import read,sha,write_json,score_boxes,mean_table,paired_stats,cost_summary

def main(trial,output,names):
 trial=Path(trial).resolve();output=Path(output).resolve()
 assert trial.parent==(ROOT/'outputs').resolve() and output.parent==(ROOT/'results').resolve()
 seal=read(trial/'sealed.json');lock=read(trial/'lock.json');assert seal['queries']==64 and seal['GT_read'] is False
 assert seal['lock_sha256']==sha(trial/'lock.json') and seal['fingerprint']==lock['fingerprint']
 records=[json.loads(s)for s in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if s.strip()]
 records=[r for r in records if r['dataset']=='hc2'][:64];assert set(seal['files'])=={r['id']for r in records}
 oldroot=ROOT/'outputs/layout_reward_dev';oldseal=read(oldroot/'scope_hc2/sealed.json');receipts={}
 for r in records:
  key=r['id'];e=seal['files'][key];p=trial/e['path'];assert p.resolve().is_relative_to(trial)
  rr=read(p.with_suffix('.json'));assert rr['GT_read'] is False and rr['fingerprint']==seal['fingerprint']
  assert sha(p)==e['sha256']==rr['sha256'] and sha(p.with_suffix('.json'))==e['receipt_sha256']
  assert sha(trial/rr['trace_path'])==rr['trace_sha256']
  assert sha(trial/rr['evidence_path'])==rr['evidence_sha256']
  op=oldroot/oldseal['files'][key]['path'];assert sha(op)==rr['reference_sha256']==oldseal['files'][key]['sha256']
  with np.load(p)as a,np.load(op)as b:
   for k in ['frame_ids','positions8','Strong_interval']:assert np.array_equal(a[k],b[k])
   for name in names:assert a[name].shape==b['E_Ref'].shape and np.isfinite(a[name]).all()and(a[name][:,2:]>0).all()
  receipts[key]=rr
 # Only now read labels; configuration selection never uses per-query GT.
 lp=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(lp)
 assert labelsha==read(oldroot/'scope_hc2/metrics.json')['labels_sha256']
 labels={r['id']:r for r in [json.loads(s)for s in lp.read_text().splitlines()if s.strip()]if r['dataset']=='hc2'}
 allnames=['ERefStrong','GeometryOPD']+names;rows=[];err=0.;checks=0
 for i,r in enumerate(records):
  key=r['id'];g=labels[key];truth={int(k):v for k,v in g['boxes'].items()}
  with np.load(trial/seal['files'][key]['path'])as a,np.load(oldroot/oldseal['files'][key]['path'])as b:
   values={}
   for name in allnames:
    boxes=b['E_Ref']if name=='ERefStrong'else b['GeometryOPD']if name=='GeometryOPD'else a[name]
    values[name],e,support=score_boxes(boxes,r,truth,a['Strong_interval'].tolist(),g['interval'],a['positions8'].tolist());err=max(err,e);checks+=1
  rows.append(dict(id=key,parent=g['parent'],values=values,support=support))
 assert len({r['parent']for r in rows})==64 and sha(lp)==labelsha
 for key,e in seal['files'].items():assert sha(trial/e['path'])==e['sha256']
 means,counts=mean_table(rows,allnames)
 proof=dict(fingerprint=seal['fingerprint'],seal_sha256=sha(trial/'sealed.json'),lock_sha256=sha(trial/'lock.json'),labels_sha256=labelsha,all64_validated_before_labels=True,official_dense_checks=checks,official_dense_max_absolute_error=err,GT_used_for_online_selection=False)
 result=dict(status='complete_HC64_exposed_development',queries=64,parents=64,means=means,valid_query_counts=counts,paired=paired_stats(rows,allnames),proof=proof,arm_order=allnames,config={k:v for k,v in lock.items()if k!='ids'})
 if all('costs'in v for v in receipts.values()):result['costs']=cost_summary(receipts,[n for n in names if n!='InitialModes'])
 write_json(trial/'metrics.json',{**result,'rows':rows})
 result['queries_anonymous']=[dict(query=f'H{i+1:03d}',values=r['values'])for i,r in enumerate(rows)]
 output.mkdir(parents=True,exist_ok=True);write_json(output/'hc64.json',result)
 lines=['# HC64 '+trial.name,'','Same fixed64 exposed development parents, one query each; Strong intervals unchanged. Same Source correspondence reward and content-knot transport are shared by all new methods; sparse/batched engineering and256 evaluation allowance, no new modes or controls. All Direct controls restore interlayer reference gradients. InitialModes is the common three-start no-adaptation readout.','', '| Method | vIoU % | Observed sIoU % | Unobserved sIoU % |','|---|---:|---:|---:|']
 for n,m in means.items():lines.append(f'| {n} | {100*m["m_vIoU"]:.4f} | {100*m["m_sIoU_observed_actual"]:.4f} | {100*m["m_sIoU_unobserved_actual"]:.4f} |')
 (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
 write_json(trial/'score_receipt.json',dict(status='scored',metrics_sha256=sha(trial/'metrics.json'),proof=proof))
 print(json.dumps(dict(status='scored',means=means)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trial',required=True);p.add_argument('--output',required=True);p.add_argument('--names',nargs='+',required=True);a=p.parse_args();main(a.trial,a.output,a.names)
