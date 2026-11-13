"""Postseal soft-support coverage diagnostic, not identity supervision."""
from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parents[2];H=R/'research/round89';B=R/'crossplatform/round89'
def read(p):return json.loads(p.read_text())
assert (B/'prediction336_attempt001/SEALED.json').exists();rows=read(H/'experiment/DEV336_PRIVATE.json');out=[]
for n,r in enumerate(rows):
 a=np.load(B/'target_evidence336_attempt001/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False);p=np.load(B/'prediction336_attempt001/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False);roles=a['token_roles'];rr={'row':'DEV'+str(n+1).zfill(4),'platform':r['platform'],'roles':len(roles),'global_hypothesis_n':int((a['hypotheses_PRIVATE']>=0).sum()),'soft_floor':.05,'not_instance_identity_accuracy':True}
 if len(roles) and roles[0].any():
  A=a['native_role_attention'][roles[0]].mean(0);A=A/max(A.sum(),1e-12);m=a['own_support']@A;ratio=m/np.maximum(m+.05*(1-m),1e-12);rr.update(own_points_mean=float(a['own_support'].sum(-1).mean()),own_attention_mass_mean=float(m.mean()),head_averaged_proxy_soft_support_share_mean=float(ratio.mean()),fraction_candidates_proxy_support_share_below_half=float((ratio<.5).mean()))
  for arm in ['frozen_bi','candidate_frozen','candidate_tta1']:
   q=int(p['queries'][p['arm_names'].tolist().index(arm)]);rr[arm+'_own_points']=int(a['own_support'][q].sum());rr[arm+'_own_attention_mass']=float(m[q]);rr[arm+'_proxy_soft_support_share']=float(ratio[q])
 out.append(rr)
(H/'SOFT_SUPPORT_COVERAGE_ROWS.json').write_text(json.dumps(out,indent=2)+'\n');print({'rows':len(out),'descriptive_proxy_not_exact_perhead_pooled_share':True})
