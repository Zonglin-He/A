"""Observation-only support eligibility; no task GT or parameter changes."""
from core88 import *
rows=[]
for i,r in enumerate(read(H88/'experiment/DEV336_PRIVATE.json')):
 e=load(B88/'evidence336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));z=prefix(r);inside=(np.abs(z['xyz'][0][None]-z['boxes'][:,None,:3])<=z['boxes'][:,None,3:]/2).all(-1).any(0)
 for j,tm in enumerate(e['token_roles']):
  if not tm.any():continue
  a=e['native_role_attention'][tm].mean(0).astype(float);a/=a.sum();reg=e['supports'][j].any(0);rows.append({'row':'DEV'+str(i+1).zfill(4),'platform':r['platform'],'role_kind':'subject' if j==0 else 'reference','role_number':j,'global_attention_mass_in_any_A256':float(a[inside].sum()),'global_attention_mass_in_three_hypotheses':float(a[reg].sum()),'three_hypothesis_point_fraction':float(reg.mean()),'native_attention_entropy':float(-(a*np.log(a.clip(1e-300))).sum()),'not_identity_purity':True})
dump(H88/'GLOBAL_SUPPORT_COVERAGE_ROWS.json',rows);print({k:float(np.median([r[k] for r in rows])) for k in ['global_attention_mass_in_any_A256','global_attention_mass_in_three_hypotheses','three_hypothesis_point_fraction']})
