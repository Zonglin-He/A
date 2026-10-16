"""Finish metadata-only binding after already saved sensitivity geometry.

No model execution, selection change or repeat GT scoring.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_referent_evidence_p0_v1 import BASE,read,put,sha
def main():
    v=BASE/'valid_content';assert (v/'RESPONSE_POSTSEAL_DIAGNOSIS.json').exists()and not(v/'DESIGN_LOCK.json').exists()
    recovery=BASE/'recovery/padding_metadata002'
    put(recovery/'FAILURE_CAPTURE.json',dict(status='captured',error_type='FileNotFoundError',scope='variant geometry already saved, missing exact parent DESIGN_LOCK citation copy; no model or selection failure',source_sha256=sha(ROOT/'scripts/diagnose_stvg_referent_padding_v1.py'),GT_read=True,actual_model_reruns=0))
    (v/'DESIGN_LOCK.json').write_bytes((BASE/'DESIGN_LOCK.json').read_bytes());assert sha(v/'DESIGN_LOCK.json')==sha(BASE/'DESIGN_LOCK.json')
    d=read(v/'RESPONSE_POSTSEAL_DIAGNOSIS.json');ss={(x['dataset'],x['strategy'],x['metric']):x for x in d['summary']};gate=[]
    for ds in ('hc2','vidstg'):
        o=ss[ds,'consensus','target_occupancy'];delta=o['mean']-max(ss[ds,s,'target_occupancy']['mean']for s in ('query','caption'));gap=ss[ds,'connection','adjacent_supported_pair_coverage']['mean']-ss[ds,'query','adjacent_supported_pair_coverage']['mean'];gate.append(dict(dataset=ds,known_sources=o['valid'],consensus_minus_best_single=delta,connection_minus_query=gap,pass_gate=o['valid']>=4 and delta>0 and gap>0))
    put(v/'PHASE2_DECISION.json',dict(status='posthoc_sensitivity_only_not_a_new_expansion_gate',count=16,gate=gate,GT_read=True,original_main_decision_unchanged=True,rule_sha256=sha(v/'DESIGN_LOCK.json'),new_AB_C_experiments_executed=False))
    put(v/'RESPONSE_CPU_COMPLETION.json',dict(status='complete',count=16,GT_read=True,result_sha256=sha(v/'RESPONSE_POSTSEAL_DIAGNOSIS.json')))
    put(v/'SENSITIVITY_COMPLETION.json',dict(status='complete',count=16,posthoc=True,GT_read=True,does_not_replace_primary_gate=True,main_decision_sha256=sha(BASE/'PHASE2_DECISION.json'),variant_result_sha256=sha(v/'RESPONSE_POSTSEAL_DIAGNOSIS.json'),scope='valid-content implementation sensitivity; CPU metadata repair only, zero repeat model/selection/GT scoring'))
    put(recovery/'RECOVERY_COMPLETION.json',dict(status='complete',count=16,GT_read=True,source_script_sha256=sha(Path(__file__)),actual_model_reruns=0,actual_extra_GT_scores=0))
    print(json.dumps(gate))
if __name__=='__main__':main()
