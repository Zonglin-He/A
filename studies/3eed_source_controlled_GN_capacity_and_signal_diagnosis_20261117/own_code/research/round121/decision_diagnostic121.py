"""Postseal source-controlled decision equivalence/rank evidence; no new model or GT."""
from pathlib import Path
import json,numpy as np,hashlib,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round121';B=R/'crossplatform/round121';rd=lambda p:json.loads(Path(p).read_text());rows=[];pairs=[('RGB','wrong_RGB'),('D','wrong_D'),('RGB','D'),('P','context_zero'),('oracle1','context_zero'),('oracle10','context_zero')];counts={str(a)+'__'+str(b):0 for a,b in pairs};rank={k:[] for k in ['context_zero','oracle1','oracle10']};missrank={k:[] for k in rank};raw=[]
for e in rd(H/'SOURCE_CACHE_REGISTRY_PRIVATE.json'):
 src=e['source'];sid=e['row']['id'];a=np.load(B/'formal'/src/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(a['arm_names'],a['scores']));qs={k:int(x.argmax()) for k,x in p.items()};y=np.load(R/e['files']['label']['path'])['ious'];valid=y.max()>0;best=np.isclose(y,y.max(),rtol=0,atol=1e-8)
 for aa,bb in pairs:counts[aa+'__'+bb]+=qs[aa]!=qs[bb]
 if valid:
  for k in rank:
   rb=int((p[k]>p[k][best].max()).sum())+1;rank[k].append(rb)
   if not best[qs['context_zero']]:missrank[k].append(rb)
 rec=rd(B/'formal'/src/'records_PRIVATE'/(sid+'.json'));raw.append(rec['endpoints']['oracle10']['actual_writeback_L2']/rec['radius'])
summary={'n':384,'no_new_model_or_GT':True,'query_difference_counts':counts,'best_IoU_candidate_ranks':{k:{'n':len(v),'mean':float(np.mean(v)),'median':float(np.median(v)),'best_selected':sum(x==1 for x in v)} for k,v in rank.items()},'initial_not_best_subset_ranks':{k:{'n':len(v),'mean':float(np.mean(v)),'median':float(np.median(v)),'best_selected':sum(x==1 for x in v)} for k,v in missrank.items()},'oracle10_cumulative_patch_to_single_radius':{'mean':float(np.mean(raw)),'median':float(np.median(raw)),'max':float(np.max(raw))},'GT_oracle_diagnostic_NOT_TTA':True};(H/'DECISION_EQUIVALENCE.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)
