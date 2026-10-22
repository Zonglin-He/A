# VidSTG original128 LLaVA-ST spatial development selection

Complete128 predictions for every scored profile, separate offline scoring; all20 signed configurations retained. The predeclared finite sequential grid selected gamma1/admission.05/guard.025/link.5 by equal-query m_vIoU. Time stays native and identical Source/Ours. This is an exposed development panel, not independent confirmation or full evaluation.

{
  "status": "complete",
  "dataset": "vidstg",
  "queries": 128,
  "parameters": {
    "gamma": 1.0,
    "admission": 0.05,
    "guard": 0.025,
    "link": 0.5
  },
  "means": {
    "Source": {
      "v": 13.916442089209461,
      "s": 27.238890815448112,
      "t": 45.98388856108531
    },
    "Ours": {
      "v": 18.490422362823587,
      "s": 33.160144486147445,
      "t": 45.98388856108531
    }
  },
  "unobserved_sIoU": {
    "Source": 27.216831570263935,
    "Ours": 32.819940255809556
  },
  "observed_sIoU": {
    "Source": 27.53022940717135,
    "Ours": 49.72676545756599
  },
  "invalid_native_outputs_retained": 0,
  "metric_denominators": {
    "Source": {
      "main": 128,
      "unobserved": 128,
      "observed": 103
    },
    "Ours": {
      "main": 128,
      "unobserved": 128,
      "observed": 103
    }
  },
  "loss_over5pp_queries": 12,
  "prediction_seal_sha256": "189c277728495eaef88a8936d39da2b2d3e79d8618d66b9ddfe4d974fd932020",
  "fixed_native_time": true,
  "no_independent_unexposed_test_claim": true
}

Post-selection descriptive paired parent bootstrap (5000 draws, seed41); these intervals do not correct development selection or prior exposure:

{
  "v": {
    "queries": 128,
    "parents": 128,
    "mean_pp": 4.573980273614124,
    "CI95_pp": [
      1.4085030764336872,
      7.655064258299566
    ],
    "equal_parent_mean_pp": 4.573980273614124,
    "wins": 65,
    "losses": 27,
    "ties": 36,
    "below_minus5": 12
  },
  "s": {
    "queries": 128,
    "parents": 128,
    "mean_pp": 5.921253670699331,
    "CI95_pp": [
      0.9298503041028509,
      10.617493190303742
    ],
    "equal_parent_mean_pp": 5.921253670699331,
    "wins": 77,
    "losses": 42,
    "ties": 9,
    "below_minus5": 27
  },
  "unobserved": {
    "queries": 128,
    "parents": 128,
    "mean_pp": 5.603108685545618,
    "CI95_pp": [
      0.6243045924223354,
      10.264947147061878
    ],
    "equal_parent_mean_pp": 5.603108685545619,
    "wins": 76,
    "losses": 43,
    "ties": 9,
    "below_minus5": 28
  }
}

Spatial means improve on observed and unobserved frames; report bootstrap and tail losses without claiming confirmed stable generalization. HC2 selection remains pending; both selections must be locked before either full run.
