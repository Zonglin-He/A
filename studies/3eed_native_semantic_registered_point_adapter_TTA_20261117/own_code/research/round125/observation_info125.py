"""Prespecified postscore descriptive support/semantic-novelty audit, never a selector."""
from pathlib import Path
import sys,json,numpy as np,torch
R=Path(__file__).resolve().parents[2];H=R/'research/round125';B=R/'crossplatform/round125';rd=lambda p:json.loads(Path(p).read_text())
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112
def main():
 torch.set_num_threads(4);assert rd(H/'SCALAR_AUDIT.json')['n']==2304;reg={(e['cohort'],e['source'],e['row']['id']):e for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json')};cols=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];heads={}
 for src in ['waymo','drone','quad']:
  S=np.load(R/cols[src]['state_path']);heads[src]={k[6:]:S[k] for k in S.files if k.startswith('match.')}
 rows=[]
 for file in sorted((H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json')):
  for r in rd(file):
   e=reg[(r['cohort'],r['source'],r['id_PRIVATE'])];row={k:r[k] for k in ['cohort','source','target','id_PRIVATE','protected_alias_fallback','B_active','C_available']};row['task_delta_IoU']=r['arms']['corr3_F']['IoU']-r['arms']['mean_context']['IoU']
   row['selected_query_changed']=r['arms']['corr3_F']['query_PRIVATE']!=r['arms']['mean_context']['query_PRIVATE']
   if not r['protected_alias_fallback']:
    sample=np.load(B/'formal'/r['cohort']/(r['source']+'_to_'+r['target'])/'samples_PRIVATE'/(r['id_PRIVATE']+'.npz'));boxes=sample['source_boxes'];a=boxes[r['arms']['mean_context']['query_PRIVATE']].astype(float);b=boxes[r['arms']['corr3_F']['query_PRIVATE']].astype(float);sa=np.maximum(a[3:6],1e-6);sb=np.maximum(b[3:6],1e-6);inter=np.maximum(0,np.minimum(a[:3]+sa/2,b[:3]+sb/2)-np.maximum(a[:3]-sa/2,b[:3]-sb/2)).prod();row['selected_pair_AABB_IoU']=float(inter/max(sa.prod()+sb.prod()-inter,1e-300));row['selected_pair_normalized_center_distance']=float(np.linalg.norm(a[:3]-b[:3])/max(np.linalg.norm((sa+sb)/2),1e-12))
    D=B/'formal'/r['cohort']/(r['source']+'_to_'+r['target']);ff=np.load(D/'features_PRIVATE'/(r['id_PRIVATE']+'.npz'));z=np.load(R/e['prefix']);obs=np.load(R/e['observations']);p='real_' if r['cohort']=='DEV' else '';bv=obs[p+'B_valid'] if p+'B_valid' in obs else np.zeros(1024,bool);cv=obs[p+'C_valid'] if p+'C_valid' in obs else np.zeros(1024,bool);ix=z['query_points_sample_inds'].astype(int).reshape(-1);mask=z['token_mask'];late=np.zeros_like(ff['fixed_late']);res=np.zeros_like(ff['fixed_r']);pa=cpu112(ff['teacher_A_h'],ff['teacher_A_u'],mask,late,res,heads[r['source']]);pb=cpu112(ff['teacher_B_h'],ff['teacher_B_u'],mask,late,res,heads[r['source']]);T=(pa+pb)/2;js=float(((pa*(np.log(pa)-np.log(T))).sum()+(pb*(np.log(pb)-np.log(T))).sum())/2);q0=r['arms']['mean_context']['query_PRIVATE'];row.update(B_seed_valid_fraction=float(bv.mean()),C_seed_valid_fraction=float(cv.mean()),B_native_probability_weighted_anchor_support=float(pa@bv[ix]),C_native_probability_weighted_anchor_support=float(pa@cv[ix]),B_support_at_zero_winner_anchor=bool(bv[ix[q0]]),B_support_at_nativeA_winner_anchor=bool(bv[ix[int(pa.argmax())]]),native_AB_JS=js,native_AB_probability_L1=float(abs(pa-pb).sum()),native_AB_same_Top1=bool(pa.argmax()==pb.argmax()))
   rows.append(row)
 summary=[]
 for co in ['DEV','EXP']:
  cr=[r for r in rows if r['cohort']==co];alln=len(cr);valid=[r for r in cr if not r['protected_alias_fallback']];active=[r for r in valid if r['B_active']];entry={'cohort':co,'all_denominator':alln,'nonalias':len(valid),'active_AB':len(active),'claim_boundary':'Query seed-anchor validity and source-distribution changes are approximate support diagnostics, NOT object visibility/identity or correctness; no cutoff chosen'}
  for field in ['B_seed_valid_fraction','B_native_probability_weighted_anchor_support','C_native_probability_weighted_anchor_support','native_AB_JS','native_AB_probability_L1']:
   vals=[r[field] for r in active];entry[field+'_mean']=float(np.mean(vals));entry[field+'_quantiles_0_25_50_75_100']=[float(v) for v in np.quantile(vals,[0,.25,.5,.75,1])]
  entry['active_native_AB_same_Top1']=sum(r['native_AB_same_Top1'] for r in active);entry['by_zero_winner_anchor_B_support']=[]
  for flag in [False,True]:
   part=[r for r in active if r['B_support_at_zero_winner_anchor']==flag];delta=[r['task_delta_IoU'] for r in part];entry['by_zero_winner_anchor_B_support'].append({'B_anchor_supported':flag,'n':len(part),'task_gain_pp_within_subgroup':float(np.mean(delta)*100) if delta else None,'contribution_pp_original_full_denominator':float(sum(delta)/alln*100),'better':sum(v>1e-12 for v in delta),'worse':sum(v< -1e-12 for v in delta)})
  entry['changed_query_geometry']=[]
  for label,accept in [('high_overlap_ge0.7',lambda r:r['selected_pair_AABB_IoU']>=.7),('low_overlap_le0.1',lambda r:r['selected_pair_AABB_IoU']<=.1),('intermediate_overlap',lambda r:.1<r['selected_pair_AABB_IoU']<.7)]:
   part=[r for r in valid if r['selected_query_changed'] and accept(r)];delta=[r['task_delta_IoU'] for r in part];entry['changed_query_geometry'].append({'geometric_group_NOT_identity_label':label,'n':len(part),'contribution_pp_original_full_denominator':float(sum(delta)/alln*100),'better':sum(v>1e-12 for v in delta),'worse':sum(v< -1e-12 for v in delta)})
  summary.append(entry)
 (H/'OBSERVATION_INFORMATION_ROWS_PRIVATE.json').write_text(json.dumps(rows,indent=2)+'\n');(H/'OBSERVATION_INFORMATION_SUMMARY.json').write_text(json.dumps({'state':'POSTSCORE_DESCRIPTIVE_ONLY_NO_GATE_OR_SELECTOR','groups':summary,'source_GT_used':False,'used_for_method_selection':False,'FINAL_access':False},indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':main()
