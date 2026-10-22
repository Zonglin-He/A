"""Offline teacher signal audit. This module reads labels; inference must not import it."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
import numpy as np,torch
from scipy.stats import spearmanr
from research.joint_event_v1.run import records,put,sha,B
from research.joint_event_v1.core import relative_event
D=R/'outputs/event_presence_hc64_v1';P=R/'results/event_presence_hc64_v1'
def auc(y,x):
 a=np.asarray(x)[y];b=np.asarray(x)[~y]
 return float(((a[:,None]>b)+.5*(a[:,None]==b)).mean()) if len(a) and len(b) else None
def corr(x,y):
 return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else None
def main():
 torch.set_num_threads(1);lab=R/'data/evaluation/development_labels.jsonl';truth={x['id']:x for x in map(json.loads,lab.read_text().splitlines())if x['dataset']=='hc2'}
 roles=json.load(open(R/'outputs/joint_event_hc64_v1/postseal_role_groups.json'));rows=[];private=[]
 for row in records():
  k=row['id'];s=torch.load(R/'outputs/event_query_hc64_v1/semantics'/f'{k}.pt',weights_only=False);c=torch.load(B/'captures'/f'{k}.pt',weights_only=False)
  v=roles[k]['predicate_groups'];n=roles[k]['nominal_groups'];z=s['premax'].numpy().astype(float);j=s['selected_query'];gt=truth[k];ids=np.asarray(c['frame_ids'])[c['planned_positions']];y=(ids+.5>=gt['interval'][0])&(ids+.5<gt['interval'][1])
  if not v or not n:continue
  verb=np.stack([z[:,j,g].mean(-1)for g in v],1).mean(1);noun=np.stack([z[:,j,g].mean(-1)for g in n],1).mean(1);rel=relative_event(z,v,n,j).mean(1);obj=np.asarray(s['native_scores']).ravel()
  design=np.c_[np.ones(8),noun];res=verb-design@np.linalg.lstsq(design,verb,rcond=None)[0]
  # High noun response remains a proxy for target presence, not visibility GT.
  high=noun>=np.median(noun)
  r=dict(query=k,event_anchors=int(y.sum()),verb_noun_correlation=corr(verb,noun),verb_noun_r2=1-float(res@res)/max(float(((verb-verb.mean())**2).sum()),1e-30),event_auc={a:auc(y,x)for a,x in dict(verb=verb,noun=noun,relative=rel,object=obj,verb_residual=res).items()},high_noun_auc=auc(y[high],verb[high]))
  rows.append(r);private.append(dict(**r,caption=row['metadata']['caption'],anchor_ids=ids.tolist(),event=y.tolist(),verb=verb.tolist(),noun=noun.tolist(),relative=rel.tolist(),object=obj.tolist(),GT_interval=gt['interval']))
 def stats(key):
  a=np.array([r[key]for r in rows if r[key]is not None]);return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)))
 summary=dict(queries=64,scope='offline exposed HC64 teacher-response audit; no inference selection from GT',labels_sha256=sha(lab),verb_noun_correlation=stats('verb_noun_correlation'),verb_noun_r2=stats('verb_noun_r2'),high_noun_auc=stats('high_noun_auc'),event_auc={a:dict(n=sum(r['event_auc'][a]is not None for r in rows),mean=float(np.mean([r['event_auc'][a]for r in rows if r['event_auc'][a]is not None])))for a in rows[0]['event_auc']},rows=rows)
 put(D/'teacher_audit_private.json',private);put(P/'teacher_audit.json',summary)
 print(json.dumps({k:v for k,v in summary.items()if k!='rows'},indent=2))
 for r in private:
  if r['query']in ['RC009','RC016','RC042']:print(json.dumps(r))
if __name__=='__main__':main()
