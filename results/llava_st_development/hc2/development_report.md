# HC2 original128 LLaVA-ST spatial development selection

All21 signed profiles retained; complete128 predictions and independent offline scoring for every candidate. The fixed sequential grid selects gamma.5/admission.05/guard.1/link.5 by equal-query m_vIoU. Source/Ours share native time; exposed development population, no independent confirmation or full evaluation claim.

{
  "status": "complete",
  "dataset": "hc2",
  "queries": 128,
  "parameters": {
    "gamma": 0.5,
    "admission": 0.05,
    "guard": 0.1,
    "link": 0.5
  },
  "means": {
    "Source": {
      "v": 9.56349098469943,
      "s": 36.5725845800253,
      "t": 23.37119775042286
    },
    "Ours": {
      "v": 12.745926638674376,
      "s": 41.88984170642389,
      "t": 23.37119775042286
    }
  },
  "unobserved_sIoU": {
    "Source": 36.512383209707735,
    "Ours": 41.64552597958236
  },
  "observed_sIoU": {
    "Source": 39.94969977691161,
    "Ours": 53.046887766613224
  },
  "invalid_native_outputs_retained": 0,
  "metric_denominators": {
    "Source": {
      "main": 128,
      "unobserved": 128,
      "observed": 123
    },
    "Ours": {
      "main": 128,
      "unobserved": 128,
      "observed": 123
    }
  },
  "loss_over5pp_queries": 7,
  "prediction_seal_sha256": "6347942daf85bdbfc109dd823887a0246176f3db799fb43038936788245729cf",
  "fixed_native_time": true,
  "no_independent_unexposed_test_claim": true
}

Post-selection descriptive parent bootstrap (5000 draws,seed41); no correction for parameter selection or prior exposure:

{
  "v": {
    "queries": 128,
    "parents": 128,
    "mean_pp": 3.1824356539749457,
    "CI95_pp": [
      1.4362010415428774,
      5.0600294517568445
    ],
    "equal_parent_mean_pp": 3.1824356539749465,
    "wins": 76,
    "losses": 28,
    "ties": 24,
    "below_minus5": 7
  },
  "s": {
    "queries": 128,
    "parents": 128,
    "mean_pp": 5.317257126398591,
    "CI95_pp": [
      2.5642147329891496,
      8.179584230496253
    ],
    "equal_parent_mean_pp": 5.31725712639859,
    "wins": 94,
    "losses": 34,
    "ties": 0,
    "below_minus5": 19
  },
  "unobserved": {
    "queries": 128,
    "parents": 128,
    "mean_pp": 5.133142769874623,
    "CI95_pp": [
      2.4058237233299273,
      7.9825401800158575
    ],
    "equal_parent_mean_pp": 5.133142769874623,
    "wins": 94,
    "losses": 34,
    "ties": 0,
    "below_minus5": 20
  }
}

Both domain selections now locked before any full. Fulls execute in VidSTG then HC2 order; no tuning from full feedback.
