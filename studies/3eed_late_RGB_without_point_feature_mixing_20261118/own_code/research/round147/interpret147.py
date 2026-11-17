"""Postseal query equality and fixed-path interaction; no new prediction or GT query."""
from common147 import *
import numpy as np

def main():
 assert rd(H/'SCALAR_AUDIT.json')['status']=='PASS';rr=rd(H/'IMAGE_ALL_ROWS_PRIVATE.json');keys={(r['source'],r['target']) for r in rr};den={k:sum(r['sampling_weight'] for r in rr if (r['source'],r['target'])==k) for k in keys};pairs=[('raw_Apoints_jointB_vs_originalA','pa_raw_joint','Frozen_A'),('normalized_jointB_vs_normalized_Aimage','pa_zero','pa_Aimage'),('normalized_jointB_vs_wrong_binding','pa_zero','pa_wrong_joint'),('position_effect_Apoints','pa_zero','pa_raw_joint'),('position_effect_mixed_points','bf_zero','bf_raw_joint'),('Apoints_vs_mixed_after_position','pa_zero','bf_zero')];rows=[]
 for label,a,b in pairs:
  ss=[r for r in rr if r['backfilled']];diff=np.array([r['IoUs'][a]-r['IoUs'][b] for r in ss]);cc=[d*r['sampling_weight']/den[r['source'],r['target']]*100/6 for d,r in zip(diff,ss)];qc=sum(r['queries_PRIVATE'][a]!=r['queries_PRIVATE'][b] for r in ss);rows.append({'comparison':label,'method':a,'control':b,'n':len(ss),'query_changed':qc,'IoU_better':int((diff>1e-12).sum()),'IoU_worse':int((diff< -1e-12).sum()),'whole_cohort_weighted_contribution_pp':float(sum(cc))})
 out={'status':'POSTSEAL_DESCRIPTIVE_ONLY','GT_scores_or_models_not_rerun':True,'does_not_choose_outputs_or_parameters':True,'all1798_denominators_preserved':True,'rows':rows,'unused_B_interpolation_zero_fields_in_inherited_auditor_are_placeholders_not_new_checks':True,'actual_new_input_check':'allPointAdapterforward_input_hashesmatchoriginalA, no newBbackbone interpolation feedsreader'};dump(H/'POSTSEAL_INTERACTION.json',out);section='\n\n## Postseal query equality and interaction\n\n'+json.dumps(out,indent=2)+'\n\nThe R146 sequentialpath doesnot imply removing pointmixturemust improve thefinalreader: the positioneffect changes withthatentry. This fixedApoint-versusmixedpoint comparison checks thisinteraction. EqualIoUmeans do notbythemselves implyequalqueries; actual querychangesarelistedabove. No newruntimegate orGT-selectedoutput isintroduced. R147 doesnotre-runBPointNet/interpolation; inheritedunusedzero auditfields arenotnew checks. The applicableinputproof is the actualPointAdapterA-inputhashcheck.\n';p=H/'REPORT.md';p.write_text(p.read_text()+section);O=H/'publication/public_export_attempt001';(O/'REPORT.md').write_text(p.read_text());dump(O/'POSTSEAL_INTERACTION.json',out);q=O/'own_code/research/round147/interpret147.py';q.write_text(Path(__file__).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));tokens=set()
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  for ob in [e['row'],e['A_sensor']]+([e['pair_PRIVATE']['B']] if e['has_B'] else []):
   for k in ['id','caption','scene','imagepath','pointpath','metapath']:
    v=ob.get(k)
    if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 for p in O.rglob('*'):
  if p.is_file():assert 'LOCAL_HOME/' not in p.read_text() and not any(t in p.read_text() for t in tokens),p
 dump(H/'POSTSEAL_EXPORT_AUDIT.json',{'status':'PASS','private_literals_scanned':len(tokens),'all_public_files_rescanned':True});dump(O/'POSTSEAL_EXPORT_AUDIT.json',rd(H/'POSTSEAL_EXPORT_AUDIT.json'));dump(O/'MANIFEST.json',{'prefix':'studies/3eed_late_RGB_without_point_feature_mixing_20261118','files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']});print(out,flush=True)
if __name__=='__main__':main()
