"""Finalize administrative primary precedence, preserving all scored evidence.

No prediction, metric, method configuration or adoption comparison changes.
The scoring utility's nested ternary prioritized the additional ablation over
the unchanged predeclared primary when both passed. Original primary wins by
design precedence; this is not argmax target-metric selection.
"""
from common134 import *
import sys,copy
def main():
 assert rd(H134/'PUBLIC_EXPORT_AUDIT.json')['n']==1798
 d=rd(H134/'DECISION.json');plan=rd(H134/'experiment/DESIGN_LOCK.json');extension=rd(H134/'experiment/TASK_RELEVANCE_EXTENSION_LOCK.json');assert d['primary']==plan['primary']==extension['original_primary_unchanged'].split(';')[0]
 snapshot=H134/'attempts/DECISION_AS_SCORED.json';dump(snapshot,d)
 selected=('R134_partial_pair3_development_candidate' if d['recipe_retained'] else ('R134_tasklinked_pair3_development_candidate' if d['tasklinked_recipe_retained'] else 'R132_originalB_LOG_partial3'))
 corrected=copy.deepcopy(d);corrected['scoring_utility_provisional_working_version']=d['current_working_version'];corrected['current_working_version']=selected;corrected['registration_rule']='original predeclared primary precedes additional ablation when both gates pass; no maximum-target-score selection'
 assert {k:v for k,v in corrected.items() if k not in ['current_working_version','scoring_utility_provisional_working_version','registration_rule']}=={k:v for k,v in d.items() if k!='current_working_version'}
 dump(H134/'DECISION.json',corrected)
 note={'status':'ADMINISTRATIVE_PRIMARY_PRECEDENCE_CORRECTED','as_scored_utility_value':d['current_working_version'],'final_registered_value':selected,'reason':'original predeclared primary precedence; scoring utility nested ternary prioritized added ablation when both pass','prediction_metric_gate_or_method_changes':False,'target_argmax_configuration_selection':False,'original_decision_sha256':sha(snapshot),'utc':utc()};dump(H134/'REGISTRATION_NOTE.json',note)
 import report134
 report134.main()
 p=H134/'REPORT.md';p.write_text(p.read_text()+'\n## Administrative registration note\n\n'+json.dumps(note,indent=2)+'\n\nAll original scored rows and gates are unchanged. The original decision is retained locally in attempts/DECISION_AS_SCORED.json, and the provisional utility field is preserved in DECISION.json. Final registration follows original-primary precedence, not target-score argmax.\n')
 import export134
 export134.O134=H134/'publication/public_export_attempt002';export134.main();O=export134.O134
 for name in ['REGISTRATION_NOTE.json']:(O/name).write_text((H134/name).read_text())
 dump(O/'DECISION_AS_SCORED.json',d)
 p=O/'REPRODUCTION.md';p.write_text(p.read_text()+'\nAfter the initial full evaluation/public-export audit, finalize134.py corrects only the working-version registration precedence, preserves the original decision and every prediction/metric/gate, regenerates the report and public-export attempt002, then independently audits the final export. The scoring utility prioritized the added ablation when both passed; original primary precedence is restored without selecting by metric magnitude.\n')
 dump(O/'MANIFEST.json',{'prefix':'studies/3eed_retained_candidate_ordinal_TTA_repair_20261117','files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']})
 # Keep both export audits/attempts; the second is the actual public payload.
 (H134/'attempts/PUBLIC_EXPORT_AUDIT_ATTEMPT001.json').write_bytes((H134/'PUBLIC_EXPORT_AUDIT.json').read_bytes())
 import audit_public134
 audit_public134.O134=O;audit_public134.main()
 print('R134_FINAL_DISPOSITION',selected,flush=True)
if __name__=='__main__':main()
