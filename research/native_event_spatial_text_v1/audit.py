"""Separately bound postseal conditional-null and footprint audit; not online scoring."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_spatial_text_v1.run import D,P,read,put,sha,records
import numpy as np,torch

def main():
 s=read(D/'sealed.json');p=read(D/'plan.json');assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');labels=R/'data/evaluation/development_labels.jsonl';assert sha(labels)==p['labels_sha256'];gt={x['id']:x for x in map(json.loads,labels.read_text().splitlines())if x['dataset']=='hc2'};rows=[]
 for row in records():
  k=row['id'];e=s['files'][k];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');ids=np.asarray(row['metadata']['frame_ids']);a,b=np.triu_indices(len(ids),1);span0=ids[a];span1=ids[b]+1;target=gt[k]['interval'];inter=np.maximum(0,np.minimum(span1,target[1])-np.maximum(span0,target[0]));true=inter/(span1-span0+target[1]-target[0]-inter);stats=tr['mask_statistics'];expect={name:float((q[a,b]*true).sum())for name,q in tr['intervened_q'].items()}
  rows.append(dict(query=k,expected_true_tIoU=expect,mean_target_control_overlap=float(np.mean([x['target_control_overlap']for x in stats if x['target_control_overlap']is not None])),mean_mask_cells=float(np.mean([x['mask_cell_mass']for x in stats])),maximum_area_mismatch=max(abs(x['mask_cell_mass']-x['control_cell_mass'])for x in stats),allcovered_cells=sum(x['fully_covered']for x in stats),fallback_count=sum(tr['background_fallbacks'].values())))
 out=dict(scope='Postseal offline expected interval quality, not new prediction arms or promotion; full-event GT not event absence labels',queries=64,mean_expected_true_tIoU={name:float(np.mean([x['expected_true_tIoU'][name]for x in rows]))for name in rows[0]['expected_true_tIoU']},mean_target_control_overlap=float(np.mean([x['mean_target_control_overlap']for x in rows])),mean_mask_cells=float(np.mean([x['mean_mask_cells']for x in rows])),max_area_mismatch=max(x['maximum_area_mismatch']for x in rows),fallback_count=sum(x['fallback_count']for x in rows),rows=rows);put(P/'intervention_audit.json',out);put(D/'intervention_audit_receipt.json',dict(code_sha256=sha(Path(__file__)),seal_sha256=sha(D/'sealed.json'),labels_sha256=sha(labels),report_sha256=sha(P/'report.json'),audit_sha256=sha(P/'intervention_audit.json'),online_GT=False));print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
