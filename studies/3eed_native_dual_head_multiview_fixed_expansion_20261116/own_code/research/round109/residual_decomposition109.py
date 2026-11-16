"""Postseal pair-margin attribution only, no new full candidate readout/selector."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round109';B=R/'crossplatform/round109';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==1536;allrows=[]
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(H/'scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))
  for row in rr:
   if row['status']['protected_alias_fallback']:continue
   a=row['arms']['real_BBF_fusedABC'];b=row['arms']['real_softtoken_geometric_fusedABC'];i=a['query_PRIVATE'];j=b['query_PRIVATE']
   if i==j:continue
   sid=row['id_PRIVATE'];z=np.load(B/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(z['arm_names'],z['scores']));f=np.load(B/name/'features_PRIVATE'/(sid+'.npz'));r=f['real_fusedABC_r'];p=ps['real_softtoken_geometric_fusedABC'];margin=float(np.log(p[j])-np.log(p[i]));dr=float(r[j]-r[i]);allrows.append({'direction':name,'row_PRIVATE':sid,'delta_IoU':b['IoU']-a['IoU'],'native_pooled_new_over_old_log_margin':margin,'static_competition_margin_new_over_old':dr,'extra_half_residual_would_reverse_THIS_PAIR':margin+.5*dr<0,'NOT_global_ranking_or_identity_truth':True})
 groups=[]
 for kind in ['all_changed','improved','harmed','same_IoU_changed']:
  rr=[r for r in allrows if kind=='all_changed' or (kind=='improved' and r['delta_IoU']>1e-12) or (kind=='harmed' and r['delta_IoU']< -1e-12) or (kind=='same_IoU_changed' and abs(r['delta_IoU'])<=1e-12)];groups.append({'group':kind,'n':len(rr),'pair_might_revert_if_full_common_residual':sum(r['extra_half_residual_would_reverse_THIS_PAIR'] for r in rr),'mean_abs_current_margin':float(np.mean([abs(r['native_pooled_new_over_old_log_margin']) for r in rr])) if rr else 0.,'mean_abs_competition_margin':float(np.mean([abs(r['static_competition_margin_new_over_old']) for r in rr])) if rr else 0.})
 (H/'RESIDUAL_PAIR_ROWS_PRIVATE.json').write_text(json.dumps(allrows,indent=2)+'\n');(H/'RESIDUAL_PAIR_DIAGNOSTIC.json').write_text(json.dumps({'n':1536,'postseal_descriptive':True,'no_new_full_ranking_or_GT_score':True,'interpretation':'existing probability pool averages full BBF(base+r) with independent CE, so common r enters at half strength. Restoring coefficient1 is a competing readout hypothesis, NOT a demonstrated bug, correctness label or newmethod result. Pair reversals do not establish globalTop1.','groups':groups},indent=2)+'\n');print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
