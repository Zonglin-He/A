"""Saved-path only: isolate identity-target finite-sample mean drift. No new model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import load,read,put,sha
from scripts.run_stvg_dino_conditional_opd32_v1 import D,P,verify


def main():
    p=verify(True);r=read(P/'RESULTS_PUBLIC.json');by={x['alias']:x for x in r['rows']};rows=[]
    for rec in p['records']:
        z=load(D/'predictions'/(rec['alias']+'.pt'));e=load(ROOT/rec['prior_prediction']);scale=np.array(p['scale'][rec['dataset']]);mu0=np.array(e['estimate']['delta'])/scale
        eps=np.random.default_rng(p['seed']+int(hashlib.sha256(rec['alias'].encode()).hexdigest()[:8],16)).standard_normal((24,4))
        out=dict(alias=rec['alias'],dataset=rec['dataset'],active=z['active'],arms={})
        for arm,fit in z['fits'].items():
            rounds=fit['rounds'];acc=sum(d['accepted_anchors']for t in rounds for d in t['details']);zero=len(rounds)>0 and acc==0
            drift=fit['mean']-mu0;pred=eps.mean(0)if arm=='Fixed_sample_distill'else np.split(eps,3)[0].mean(0)+np.split(eps,3)[1].mean(0)+np.split(eps,3)[2].mean(0)if arm=='On_policy_distill'else np.zeros(4)
            error=float(abs(drift-pred).max())if zero else None
            if zero:assert error<1e-12
            out['arms'][arm]=dict(accepted_anchor_feedback=acc,all_teacher_targets_identity=zero,
                finite_sample_only_mean_drift_exact=error,student_mean_displacement_norm=float(np.linalg.norm(drift)),
                v_change_E_pp=100*(by[rec['alias']]['values'][arm]['v']-by[rec['alias']]['values']['E_MBR']['v']))
        rows.append(out)
    groups={}
    for ds in ['hc2','vidstg']:
        rr=[x for x in rows if x['dataset']==ds];groups[ds]={}
        for arm in ['One_correction','Fixed_sample_distill','On_policy_distill']:
            zero=[x for x in rr if x['arms'][arm]['all_teacher_targets_identity']];sig=[x for x in rr if x['arms'][arm]['accepted_anchor_feedback']>0]
            groups[ds][arm]=dict(active_queries=sum(x['active']for x in rr),queries_with_any_accepted_feedback=len(sig),active_queries_all_identity_targets=len(zero),
                all_identity_signed_v_change_sum_pp=float(sum(x['arms'][arm]['v_change_E_pp']for x in zero)),
                all_identity_v_contribution_to_all16_mean_pp=float(sum(x['arms'][arm]['v_change_E_pp']for x in zero)/16),
                feedback_present_v_contribution_to_all16_mean_pp=float(sum(x['arms'][arm]['v_change_E_pp']for x in sig)/16),
                note='descriptive saved paths; no claimed alternative teacher evaluation at counterfactual states')
    put(P/'NO_FEEDBACK_DRIFT_DIAGNOSIS.json',dict(status='pass_saved_path_diagnosis',count=32,groups=groups,rows=rows,
        main_limitation='Teacher identity on sampled actions does not imply distribution-level identity under empirical mean M-projection. Dynamic no-response rounds accumulate finite-sample noise.',
        no_new_model_or_candidate_or_GT_scoring=True,no_changed_predictions=True,no_after_GT_rule_change_or_rerun=True,
        interpretation='Severe dynamic loss cannot be attributed solely to teacher visual correction or on-policy feedback. This implementation fails a desirable no-feedback/no-update invariant; it is not a clean verdict against the requested mechanism.',
        current_run_stop=True))
    print(groups)


if __name__=='__main__':main()
