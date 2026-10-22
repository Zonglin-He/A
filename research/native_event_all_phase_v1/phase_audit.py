"""Post-seal middle-evidence compatibility audit; never an online acceptance rule."""
import os,sys,json
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_all_phase_v1.run import D,P,read,put,sha,records,B
from research.native_event_shared_query_v1.analyze import overlap_boxes
import numpy as np,torch

def main():
 torch.set_num_threads(1);plan=read(D/'plan.json');seal=read(D/'sealed.json');assert seal['count']==64 and not seal['GT_read'];assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');labels=R/'data/evaluation/development_labels.jsonl';assert sha(labels)==plan['labels_sha256'];truth={x['id']:x for x in map(json.loads,labels.read_text().splitlines())if x['dataset']=='hc2'};metrics={x['query']:x for x in read(P/'report.json')['rows']};rows=[]
 for row in records():
  k=row['id']
  if k not in plan['new_middle_members']:continue
  e=seal['files'][k];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');f=B/'captures'/f'{k}.pt';assert sha(f)==plan['captures'][k]['capture_sha256'];cap=torch.load(f,weights_only=False,map_location='cpu');ids=np.array(cap['frame_ids']);i,j=np.triu_indices(len(ids),1);gt=truth[k]['interval'];inside=(ids[i]>=gt[0])&(ids[j]+1<=gt[1]);overlap=(ids[i]<gt[1])&(ids[j]+1>gt[0]);phases=[]
  for name,q in tr['intervened_q'].items():
   if name=='Full':continue
   phases.append(dict(phase=name,posterior_mass_fully_inside_full_event=float(q[i,j][inside].sum()),posterior_mass_any_overlap_full_event=float(q[i,j][overlap].sum()),Source_box_IoU_vs_Full=float(overlap_boxes(tr['middle_source_boxes'][name],cap['Source']).mean())))
  r=metrics[k];rows.append(dict(query=k,middle_phases=phases,delta_t_all_vs_endpoint_pp=100*(r['values']['AllPhaseOrdered']['t']-r['values']['EndpointOrdered']['t']),delta_v_all_vs_endpoint_pp=100*(r['values']['AllPhaseOrdered']['v']-r['values']['EndpointOrdered']['v']),valid_order_mass=tr['order_mass']))
 flat=[x for r in rows for x in r['middle_phases']];assert len(rows)==21 and len(flat)==30 and not torch.cuda.is_initialized();out=dict(scope='AFTER64seal+independent score; exposed development only; no online GT/threshold/acceptance',queries=21,middle_texts=30,mean_middle_any_overlap_mass=float(np.mean([x['posterior_mass_any_overlap_full_event']for x in flat])),middle_majority_no_event_overlap=sum(x['posterior_mass_any_overlap_full_event']<.5 for x in flat),middle_Source_box_proxy_below_half=sum(x['Source_box_IoU_vs_Full']<.5 for x in flat),middle_Source_box_proxy_mean=float(np.mean([x['Source_box_IoU_vs_Full']for x in flat])),caveat='GT is FULL-description interval, not separately annotated subevents; membership is compatibility proxy, not atomic-event truth. Box overlap is conditioning/identity proxy, not semantic truth. No rejection rule or claim of causal proof.',rows=rows);put(P/'phase_evidence_audit.json',out);put(D/'phase_evidence_audit_receipt.json',dict(seal_sha256=sha(D/'sealed.json'),scored_report_sha256=sha(P/'report.json'),audit_sha256=sha(P/'phase_evidence_audit.json'),code_sha256=sha(Path(__file__)),labels_sha256=sha(labels),inference_unchanged=True,CUDA_initialized=False));print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
 for x in sorted(rows,key=lambda z:z['delta_t_all_vs_endpoint_pp'])[:4]:print(json.dumps(x))
if __name__=='__main__':main()
