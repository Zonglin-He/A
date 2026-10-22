"""Postseal binding audit uses scored decisions and Source-box identity proxies only."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_joint_caption_v1.run import D,P,read,put,sha
import numpy as np,torch
def box_iou(a,b):
 def corners(x):return np.c_[x[:,:2]-x[:,2:]/2,x[:,:2]+x[:,2:]/2]
 x=corners(a);y=corners(b);i=np.maximum(0,np.minimum(x[:,2:],y[:,2:])-np.maximum(x[:,:2],y[:,:2])).prod(1);return i/np.maximum(a[:,2:].prod(1)+b[:,2:].prod(1)-i,1e-30)
def main():
 torch.set_num_threads(1);s=read(D/'sealed.json');r=read(P/'report.json');assert s['count']==64 and not s['GT_read'];assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');rows=[];rr={x['query']:x for x in r['rows']}
 for k,e in s['files'].items():
  assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');q=tr['q'];bs=tr['variant_source_boxes'];proxies={a:float(box_iou(b,bs['Full']).mean())for a,b in bs.items()if a!='Full'}
  rows.append(dict(query=k,supported=e['supported'],source_box_IoU_vs_original=proxies,posterior_TV={a:float(np.abs(q['JointCaptionFusion']-q[a]).sum()/2)for a in ['ImplicitSubjectFusion','PredicateBindingFusion','ReverseJointFeedback','SharpenOriginal']},values=rr[k]['values'],movement=rr[k]['movement']))
 out=dict(scope='Sealed scored development. Variant Sourcebox agreement is identity/conditioning proxy, not GT or online gate; no atomic-event truth available. Tokenbag controls may describe true or malformed events.',queries=64,supported=52,source_box_proxy_mean={a:float(np.mean([x['source_box_IoU_vs_original'][a]for x in rows if x['supported']]))for a in ['Joint','Implicit','Binding']},source_box_proxy_below_half={a:sum(x['source_box_IoU_vs_original'][a]<.5 for x in rows if x['supported'])for a in ['Joint','Implicit','Binding']},mean_posterior_TV={a:float(np.mean([x['posterior_TV'][a]for x in rows]))for a in rows[0]['posterior_TV']},rows=rows)
 put(P/'binding_audit.json',out);put(D/'binding_audit_receipt.json',dict(audit_code_sha256=sha(Path(__file__)),seal_sha256=sha(D/'sealed.json'),report_sha256=sha(P/'report.json'),output_sha256=sha(P/'binding_audit.json'),new_model_calls=0,new_labels_access=0,postseal_only=True));print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
