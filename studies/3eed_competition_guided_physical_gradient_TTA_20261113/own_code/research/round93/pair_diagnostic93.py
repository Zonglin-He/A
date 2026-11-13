"""Post-seal IoU utility of every weak pair, never an identity label or update gate."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round93';P=R/'crossplatform/round93/prediction336_attempt001';rd=lambda p:json.loads(p.read_text());assert rd(H/'independent_scalar_audit_attempt001/AUDIT_RESULT.json')['n']==336
rows=rd(H/'experiment/DEV336_PRIVATE.json');Y={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in rd(R/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};out=[]
for n,r in enumerate(rows):
 e=np.load(P/'evidence_PRIVATE'/(r['id']+'.npz'));s=e['zero_base_PRIVATE']+e['zero_residual_PRIVATE'];q=int(s.argmax());y=Y[r['id']]
 for k,((i,j),t) in enumerate(zip(e['pairs_PRIVATE'],e['targets_PRIVATE'])):
  preferred,other=(int(i),int(j)) if t>0 else (int(j),int(i));a,b=float(y[preferred]),float(y[other]);out.append({'row':'DEV'+str(n+1).zfill(4),'platform':r['platform'],'pair_ordinal':k+1,'initial_margin_preferred_minus_other':float(s[preferred]-s[other]),'target_margin_preferred_minus_other':float(abs(t)),'preferred_IoU':a,'other_IoU':b,'preferred_has_higher_IoU':a>b+1e-10,'preferred_has_lower_IoU':a<b-1e-10,'IoU_tie':abs(a-b)<=1e-10,'both_zero_IoU':a==b==0,'touches_zero_top1':q in (i,j),'opposes_zero_top1':q==other,'postseal_only_not_identity_GT':True})
(H/'PAIR_PREFERENCE_DIAGNOSTIC_ROWS.json').write_text(json.dumps(out,indent=2)+'\n');print({'pairs':len(out),'higher':sum(x['preferred_has_higher_IoU'] for x in out),'lower':sum(x['preferred_has_lower_IoU'] for x in out),'bothzero':sum(x['both_zero_IoU'] for x in out)})
