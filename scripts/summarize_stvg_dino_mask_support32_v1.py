"""Explicit known-GT support alongside original-roster gated contributions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
from scripts.run_stvg_dino_mask_align32_v1 import D,P,verify
from scripts.score_stvg_joint_robust_correction32_v1 import paired


def main():
    verify();r=read(P/'RESULTS_PUBLIC.json');groups={}
    for ds,g in r['groups'].items():
        rr=[x for x in r['rows']if x['dataset']==ds];known=[x for x in rr if x['accounting']['B_aligned']['known_GT_samples']>0]
        unknown=[x['alias']for x in rr if x['active']and x['accounting']['B_aligned']['known_GT_samples']==0]
        groups[ds]=dict(original_roster_queries=16,queries_with_known_anchor_GT=len(known),active_queries_without_anchor_GT=unknown,
            unsupported_no_action_queries=[x['alias']for x in rr if not x['active']],
            known_anchor_query_quality={m:{k:paired([x['accounting'][m][k]for x in known])for k in ['gated_teacher_minus_sample_pp','gated_teacher_minus_E_pp','gated_shared_minus_E_pp','gated_compression_minus_local_pp']}for m in ['A_unaligned','B_aligned']},
            roster_contribution={m:{k:g['accounting'][m][k]for k in ['gated_teacher_minus_sample_pp','gated_teacher_minus_E_pp','gated_shared_minus_E_pp','gated_compression_minus_local_pp']}for m in ['A_unaligned','B_aligned']})
    put(P/'KNOWN_GT_SUPPORT_ACCOUNTING.json',dict(status='explicit_denominator_disclosure',count=32,groups=groups,
        meaning='RESULTS/REPORT gated fields average observed-known-anchor contributions over the original16 roster; queries without any scorable anchor contribute0 observed contribution, not0 estimated correctness. This supplement separately excludes unknown-anchor queries for actual known-support means.',
        no_missing_GT_counted_as_failure=True,no_new_GT_reads=True,outputs_full_metrics_and_decision_unchanged=True))
    print({d:dict(known=g['queries_with_known_anchor_GT'],active_unknown=g['active_queries_without_anchor_GT'],B_teacher_E=g['known_anchor_query_quality']['B_aligned']['gated_teacher_minus_E_pp']['mean'],B_shared_E=g['known_anchor_query_quality']['B_aligned']['gated_shared_minus_E_pp']['mean'])for d,g in groups.items()})


if __name__=='__main__':main()
