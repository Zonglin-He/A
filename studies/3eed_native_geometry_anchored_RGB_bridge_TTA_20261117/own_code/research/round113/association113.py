"""Postseal descriptive audit of query-seed RGB sampling versus fixed final box.
Not an instance-identity label or test-time gate; all examples retained.
"""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round113';B=R/'crossplatform/round113';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert (H/'SCALAR_AUDIT.json').exists();out=[];summary=[];primary=rd(H/'experiment/DESIGN_LOCK.json')['primary'];control='anchor_frozen_feature3_fusedABC';arms=[primary,control,'anchor_noLate_feature3_fusedABC','neither_correct_fusedABC','native_real_fusedABC']
 for f in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json'):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','')
  for r in rd(f):
   if r['protected_alias_fallback']:continue
   sid=r['id_PRIVATE'];z=np.load(R/'crossplatform/round111'/name/'prefix_PRIVATE'/(sid+'.npz'));box=z['boxes'];xyz=z['xyz'][0];ids=z['query_points_sample_inds'].astype(int);anchor=xyz[ids];within=(abs(anchor-box[:,:3])<=box[:,3:]/2).all(-1);distance=np.maximum(0,abs(anchor-box[:,:3])-box[:,3:]/2);normalized=np.max(abs(anchor-box[:,:3])/np.maximum(box[:,3:]/2,1e-6),-1);rec=rd(B/name/'records_PRIVATE'/(sid+'.json'));gain=r['arms'][primary]['IoU']-r['arms'][control]['IoU'];row={'source':r['source'],'target':r['target'],'id_PRIVATE':sid,'main_delta_IoU':gain,'all256_anchor_inside_fraction':float(within.mean()),'all256_anchor_box_distance_median':float(np.median(np.linalg.norm(distance,axis=1)))}
   for arm in arms:
    q=r['arms'][arm]['query_PRIVATE'];row[arm+'__anchor_inside_box']=bool(within[q]);row[arm+'__anchor_to_box_distance']=float(np.linalg.norm(distance[q]));row[arm+'__normalized_anchor_offset']=float(normalized[q])
   out.append(row)
 for group in ['all','waymo','drone','quad']:
  for tail in ['all','improved','harmed','unchanged']:
   rr=[r for r in out if (group=='all' or r['target']==group) and (tail=='all' or (tail=='improved' and r['main_delta_IoU']>1e-12) or (tail=='harmed' and r['main_delta_IoU']< -1e-12) or (tail=='unchanged' and abs(r['main_delta_IoU'])<=1e-12))]
   if not rr:continue
   for arm in arms:summary.append({'group':group,'tail':tail,'arm':arm,'n':len(rr),'selected_anchor_inside_finalbox_fraction':float(np.mean([r[arm+'__anchor_inside_box'] for r in rr])),'selected_anchor_to_finalbox_distance_mean':float(np.mean([r[arm+'__anchor_to_box_distance'] for r in rr])),'identity_certified':False})
 (H/'QUERY_RGB_ASSOCIATION_PRIVATE.json').write_text(json.dumps(out,indent=2)+'\n');(H/'QUERY_RGB_ASSOCIATION_SUMMARY.json').write_text(json.dumps({'postseal_descriptive_only':True,'not_a_gate_or_instance_identity_truth':True,'n_nonalias':len(out),'rows':summary},indent=2)+'\n');print('ASSOCIATION113',len(out),flush=True)
if __name__=='__main__':main()
