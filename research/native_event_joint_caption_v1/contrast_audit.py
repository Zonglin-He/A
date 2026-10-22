"""Exploratory POSTSEAL role-binding contrast diagnostic, not an online method."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_joint_caption_v1.run import D,P,read,put,sha,records
import numpy as np,torch
def main():
 torch.set_num_threads(1);seal=read(D/'sealed.json');plan=read(D/'plan.json');assert seal['count']==64 and not seal['GT_read'];assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json')
 label=R/'data/evaluation/development_labels.jsonl';assert sha(label)==plan['labels_sha256'];gt={x['id']:x for x in map(json.loads,label.read_text().splitlines())if x['dataset']=='hc2'};rows=[]
 for row in records():
  k=row['id'];e=seal['files'][k]
  if not e['supported']:continue
  assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');qq=tr['intervened_q'];q0=qq['Full'];ids=np.asarray(row['metadata']['frame_ids']);i,j=np.triu_indices(len(ids),1);take=q0[i,j]>0;i=i[take];j=j[take];span=np.c_[ids[i],ids[j]+1];length=span[:,1]-span[:,0];target=gt[k]['interval'];inter=np.maximum(0,np.minimum(span[:,1],target[1])-np.maximum(span[:,0],target[0]));truth=inter/(length+target[1]-target[0]-inter);lp=np.log(q0[i,j]);joint=np.log(qq['Joint'][i,j]);binding=np.log(qq['Binding'][i,j]);scores=dict(BindingContrast=joint-binding,TimeReverseContrast=(np.log(np.maximum(qq['Joint'],1e-300))-np.log(np.maximum(qq['Binding'],1e-300)))[::-1,::-1].T[i,j],JointLog=joint,NativeLog=lp);wins={a:0. for a in scores};n=0
  for z in range(len(i)):
   sel=np.flatnonzero((np.arange(len(i))>z)&(abs(length-length[z])<=.1*length[z])&(abs(lp-lp[z])<=.25)&(abs(truth-truth[z])>1e-10));direction=np.sign(truth[sel]-truth[z]);n+=len(sel)
   for a,v in scores.items():
    d=direction*(v[sel]-v[z]);wins[a]+=float(((d>1e-10)+.5*(abs(d)<=1e-10)).sum())
  rows.append(dict(query=k,pairs=n,accuracy={a:v/n if n else None for a,v in wins.items()},expected_true_tIoU={a:float((q[i,j]*truth).sum())for a,q in qq.items()},identical_correct_wrong_posterior=bool(np.array_equal(qq['Joint'],qq['Binding']))))
 rows=[x for x in rows if x['pairs']];ix=np.random.default_rng(20261023).integers(0,len(rows),(10000,len(rows)));arrays={a:np.array([x['accuracy'][a]for x in rows])for a in rows[0]['accuracy']};means={a:dict(mean=float(v.mean()),CI95=np.quantile(v[ix].mean(1),[.025,.975]).tolist())for a,v in arrays.items()};delta=arrays['BindingContrast']-arrays['NativeLog']
 out=dict(scope='Exploratory postseal offline matched-duration/prior rank diagnostic of genuinely SAME-BPE-bag predicate binding contrast. No new prediction method, fitting, acceptance gate or selected hyperparameters. Not the masked/nominal/null ratio already tested; counterfactual still may be true or malformed.',supported_parents=len(rows),all_queries=64,matched_pairs=sum(x['pairs']for x in rows),macros=means,contrast_minus_native=dict(mean=float(delta.mean()),CI95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist()),identical_correct_wrong_posteriors=sum(x['identical_correct_wrong_posterior']for x in rows),expected_true_tIoU={a:float(np.mean([x['expected_true_tIoU'][a]for x in rows]))for a in rows[0]['expected_true_tIoU']},rows=rows)
 put(P/'contrast_information_audit.json',out);put(D/'contrast_audit_receipt.json',dict(audit_code_sha256=sha(Path(__file__)),labels_sha256=sha(label),original_seal_sha256=sha(D/'sealed.json'),scored_report_sha256=sha(P/'report.json'),output_sha256=sha(P/'contrast_information_audit.json'),postseal_exploratory_only=True,new_model_calls=0,new_online_prediction_methods=0));print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
