"""Post-seal event-specificity and identity-proxy audit. No online GT selection."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_position_v1.run import D,P,read,put,sha,records
sys.path.insert(0,str(R/'external/TA-STVG'))
import utils.misc,utils.video_list
import numpy as np,torch

def overlap_boxes(a,b):
 def corners(x):return np.c_[x[:,:2]-x[:,2:]/2,x[:,:2]+x[:,2:]/2]
 x=corners(a);y=corners(b);inter=np.maximum(0,np.minimum(x[:,2:],y[:,2:])-np.maximum(x[:,:2],y[:,:2])).prod(1);return inter/np.maximum(a[:,2:].prod(1)+b[:,2:].prod(1)-inter,1e-30)
def corr(a,b):return float(np.corrcoef(a,b)[0,1])if np.std(a)>1e-12 and np.std(b)>1e-12 else None

def main():
 torch.set_num_threads(1);s=read(D/'sealed.json');assert s['count']==64 and not s['GT_read'];assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');plan=read(D/'plan.json')
 for e in s['files'].values():assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']
 labels=R/'data/evaluation/development_labels.jsonl';assert sha(labels)==plan['labels_sha256'];gt={x['id']:x for x in map(json.loads,labels.read_text().splitlines())if x['dataset']=='hc2'};report=read(P/'report.json');vals={x['query']:x for x in report['rows']};rows=[]
 for row in records():
  k=row['id'];e=s['files'][k];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');ids=np.asarray(row['metadata']['frame_ids']);i,j=np.triu_indices(len(ids),1);take=tr['q']['NativeG6'][i,j]>0;i=i[take];j=j[take];span=np.c_[ids[i],ids[j]+1];length=span[:,1]-span[:,0];q=tr['q']['NativeG6'][i,j];lp=np.log(q);score=tr['score'][i,j];target=gt[k]['interval'];inter=np.maximum(0,np.minimum(span[:,1],target[1])-np.maximum(span[:,0],target[0]));true=inter/(length+target[1]-target[0]-inter);win=equal=total=basewin=0
  # Actual same-duration and near-equal native-prior pairs, all eligible pairs;
  # no choosing online candidates/parameters from this diagnostic.
  for a in range(len(i)):
   sel=np.flatnonzero((np.arange(len(i))>a)&(abs(length-length[a])<=.1*length[a])&(abs(lp-lp[a])<=.25)&(abs(true-true[a])>1e-10));direction=np.sign(true[sel]-true[a]);d=direction*(score[sel]-score[a]);win+=int((d>1e-10).sum());equal+=int((abs(d)<=1e-10).sum());total+=len(sel);basewin+=float(((direction*(lp[sel]-lp[a])>1e-10)+.5*(abs(lp[sel]-lp[a])<=1e-10)).sum())
  identity={}
  r=dict(query=k,supported=True,matched_pairs=total,score_pair_accuracy=(win+.5*equal)/total if total else None,native_pair_accuracy=basewin/total if total else None,score_log_duration_corr=corr(score,np.log(length)),intervention_box_IoU_vs_full=identity,values=vals[k]['values'],movement=vals[k]['movement'])
  rows.append(r)
 macro=np.array([r['score_pair_accuracy']for r in rows if r['score_pair_accuracy']is not None]);ix=np.random.default_rng(20261023).integers(0,len(macro),(10000,len(macro)));out=dict(scope='postseal offline exposed HC64 audit; no online GT or method selection',queries=64,supported=sum(x['supported']for x in rows),matched_candidate_pair_macro_accuracy=float(macro.mean()),CI95=np.quantile(macro[ix].mean(1),[.025,.975]).tolist(),parents_with_pairs=len(macro),matched_candidate_pairs=sum(r['matched_pairs']for r in rows),native_matched_pair_macro_accuracy=float(np.mean([r['native_pair_accuracy']for r in rows if r['native_pair_accuracy']is not None])),identity_proxy_mean={},identity_proxy_below_half={},rows=rows)
 put(P/'information_audit.json',out);print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
