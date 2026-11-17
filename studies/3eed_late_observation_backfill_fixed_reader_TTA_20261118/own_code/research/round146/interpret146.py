"""Postseal component decomposition; never selects a prediction or parameter."""
from common146 import *
import numpy as np

def main():
 assert rd(H/'SCALAR_AUDIT.json')['status']=='PASS';rr=rd(H/'ALL_ROWS_PRIVATE.json');keys={(r['source'],r['target']) for r in rr};den={k:sum(r['sampling_weight'] for r in rr if (r['source'],r['target'])==k) for k in keys};pairs=[('new_point_mixture_before_newRGB_or_position','bf_raw_F_A','Frozen_A'),('new_RGB_after_point_mixture','bf_raw_joint','bf_raw_F_A'),('new_position_conditioning_after_newRGB','bf_zero','bf_raw_joint'),('new_Up_update_after_new_observation','bf_TTA','bf_zero')];rows=[]
 for label,a,b in pairs:
  cc=[(r['IoUs'][a]-r['IoUs'][b])*r['sampling_weight']/den[r['source'],r['target']]*100/6 for r in rr if r['backfilled']];rows.append({'component':label,'method':a,'control':b,'whole_cohort_weighted_contribution_pp':float(sum(cc))})
 parent=next(a['mIoU'] for a in rd(H/'AGGREGATE_ROWS.json') if a['aggregation']=='design_weighted' and a['arm']=='norm_TTA');mainv=next(a['mIoU'] for a in rd(H/'AGGREGATE_ROWS.json') if a['aggregation']=='design_weighted' and a['arm']=='bf_TTA');assert abs(sum(x['whole_cohort_weighted_contribution_pp'] for x in rows)-(mainv-parent))<1e-10
 out={'status':'PASS_POSTSEAL_DESCRIPTIVE_ONLY','new_n':166,'fixed_order_path_not_all_interactions_or_universal_causal_effect':True,'old1632_parent_predictions_exact':True,'task_GT_scoring_not_repeated':True,'new_percase_or_direction_choice':False,'rows':rows,'main_minus_parent_pp':mainv-parent};dump(H/'COMPONENT_PATH_POSTSEAL.json',out);section='\n\n## Postseal measured component path (no predictor changes)\n\n'+json.dumps(out,indent=2)+'\n\nThis is the fixed actual path on the166backfilledinputs, expressed withoriginalwholecohort designweights/denominators. The differences telescope exactlytoR146minusR145 becauseold1632predictions areunchanged. The firstcomparison usesoriginalcompleteA ontheseoriginalnoBcases. It separates rawpointfeature averaging, addedRGBbank, fixedpositionalconditioning andUpupdate alongthispath, butdoesnotmeasureallinteractions orprove badregistration/physicalobjectidentity. Allendpoints were alreadysealedandscalaraudited; no newtaskIoU query, task-basedfilter oroutputswitch isintroduced.\n';p=H/'REPORT.md';p.write_text(p.read_text()+section);O=H/'publication/public_export_attempt001';(O/'REPORT.md').write_text(p.read_text());dump(O/'COMPONENT_PATH_POSTSEAL.json',out);q=O/'own_code/research/round146/interpret146.py';q.write_text(Path(__file__).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));tokens=set()
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  for ob in [e['row'],e['A_sensor']]+([e['pair_PRIVATE']['B']] if e['has_B'] else []):
   for k in ['id','caption','scene','imagepath','pointpath','metapath']:
    v=ob.get(k)
    if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 for p in O.rglob('*'):
  if p.is_file():assert 'LOCAL_HOME/' not in p.read_text() and not any(t in p.read_text() for t in tokens),p
 dump(H/'POSTSEAL_INTERPRETATION_EXPORT_AUDIT.json',{'status':'PASS','private_literals_scanned':len(tokens),'all_public_files_rescanned':True});dump(O/'POSTSEAL_INTERPRETATION_EXPORT_AUDIT.json',rd(H/'POSTSEAL_INTERPRETATION_EXPORT_AUDIT.json'));dump(O/'MANIFEST.json',{'prefix':'studies/3eed_late_observation_backfill_fixed_reader_TTA_20261118','files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']});print(out,flush=True)
if __name__=='__main__':main()
