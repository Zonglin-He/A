# R134 disposition

{
  "primary": "partial_pair_F",
  "target_GT_configuration_selection": false,
  "FINAL_access": false,
  "comparisons": {
    "raw": {
      "mIoU_percent": 12.911249468788144,
      "gains_pp": {
        "zero_F": 0.21598678449539577,
        "partial3_F": 0.08294439155034894,
        "pair_direct_F": 0.21057726867999982,
        "partial3_pair_direct_F": 0.08945198607384341,
        "partial_wrongpair_F": 0.08711066605994944
      }
    },
    "design_weighted": {
      "mIoU_percent": 12.873988195727632,
      "gains_pp": {
        "zero_F": 0.2434697673507884,
        "partial3_F": 0.0808812928216085,
        "pair_direct_F": 0.23643894793037212,
        "partial3_pair_direct_F": 0.08738654809093305,
        "partial_wrongpair_F": 0.08379008048089531
      }
    }
  },
  "recipe_retained": true,
  "tasklinked_comparisons": {
    "raw": {
      "mIoU_percent": 12.829223439894363,
      "gains_pp": {
        "zero_F": 0.13396075560161513,
        "partial3_F": 0.0009183626565683056,
        "tasklinked_direct_F": 0.1291368632478722,
        "tasklinked_partial3_direct_F": 0.008595482648063424,
        "tasklinked_wrongpair_F": 0.013274648186326843
      }
    },
    "design_weighted": {
      "mIoU_percent": 12.794076518866378,
      "gains_pp": {
        "zero_F": 0.16355809048953418,
        "partial3_F": 0.0009696159603542753,
        "tasklinked_direct_F": 0.15711614263491924,
        "tasklinked_partial3_direct_F": 0.008643156665048224,
        "tasklinked_wrongpair_F": 0.013394434438218639
      }
    }
  },
  "tasklinked_recipe_retained": true,
  "robust_method_adopted": false,
  "independent_validation": false,
  "current_working_version": "R134_partial_pair3_development_candidate",
  "scoring_utility_provisional_working_version": "R134_tasklinked_pair3_development_candidate",
  "registration_rule": "original predeclared primary precedes additional ablation when both gates pass; no maximum-target-score selection"
}

{
  "n": 508,
  "categories": {
    "recoverable_Acc25_candidate_inside_C": 146,
    "Acc25_candidate_outside_C": 116,
    "already_Acc25_positive": 145,
    "no_Acc25_candidate_in_A256": 101
  },
  "category_with_new_pair_evidence": {
    "Acc25_candidate_outside_C": 36,
    "no_Acc25_candidate_in_A256": 46,
    "already_Acc25_positive": 49,
    "recoverable_Acc25_candidate_inside_C": 63
  },
  "category_with_top_reversal": {
    "no_Acc25_candidate_in_A256": 2,
    "recoverable_Acc25_candidate_inside_C": 3,
    "already_Acc25_positive": 1,
    "Acc25_candidate_outside_C": 1
  },
  "new_query_changed": 24,
  "tasklinked_new_query_changed": 7
}
