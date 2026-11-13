"""After source prediction seals: count original geometric proxy eligibility, never filter."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round90';read=lambda p:json.loads(p.read_text());out=[]
assert (R/'crossplatform/round90/source_aux128_attempt001/SEALED.json').exists()
for split in ['TRAIN512','VAL128']:
 rows=read(R/('research/round76/input/SOURCE_'+split+'_PRIVATE.json'));stats={'split':split,'n':len(rows),'target_oracle_above25_n':0,'target_oracle_above50_n':0,'spatial_role_available_n':0,'target_slot_proxy_eligible_n':0,'reference_available_n':0,'positive_proposal_count':0,'eligible_positive_proposal_count':0}
 for r in rows:
  with np.load(R/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as a:y=a['ious']
  with np.load(R/'crossplatform/round89/source_features640_attempt001/routes_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as a:
   E=len(a['token_roles']);stats['spatial_role_available_n']+=int(E>0);stats['reference_available_n']+=int(E>1);pos=y>.25;stats['target_oracle_above25_n']+=int(pos.any());stats['target_oracle_above50_n']+=int((y>.5).any());stats['positive_proposal_count']+=int(pos.sum())
   if E:
    ids=a['hypotheses_PRIVATE'][0];known=ids>=0;yy=np.zeros(3);yy[known]=y[ids[known]];global_valid=(yy>.25)&(a['weights_plain'][0]>0);eligible=pos&(a['own_valid']|global_valid.any());stats['target_slot_proxy_eligible_n']+=int(eligible.any());stats['eligible_positive_proposal_count']+=int(eligible.sum())
 out.append(stats)
(H/'SOURCE_PROXY_ELIGIBILITY_ROWS.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
