# hc2 / frame_freeze_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.248221189401391,
      "t": 0.4644731967076649,
      "s": 0.5202135291465216,
      "fixed_actual_sIoU": 0.524518833932415,
      "observed_actual_sIoU": 0.5250619955089314,
      "unobserved_actual_sIoU": 0.5263567981505167,
      "boundary_MAE": 82.6796875,
      "start_error_seconds": 3.115076067727464,
      "end_error_seconds": 3.1430323998785648,
      "interval_length_seconds": 6.159513532953898,
      "length_error_seconds": 3.2768838063564543,
      "vIoU_at_03": 0.421875,
      "vIoU_at_05": 0.1484375,
      "vIoU_ge_03": 0.421875,
      "vIoU_ge_05": 0.1484375,
      "tIoU_ge_05": 0.484375,
      "tIoU_eq_0": 0.1328125,
      "boundary_MAE_seconds": 3.1290542338030143
    },
    "FullMAP": {
      "v": 0.31287616601188717,
      "t": 0.5450503286061676,
      "s": 0.5486672788129363,
      "fixed_actual_sIoU": 0.5504688925432498,
      "observed_actual_sIoU": 0.5728196703090285,
      "unobserved_actual_sIoU": 0.5461457338967792,
      "boundary_MAE": 76.8203125,
      "start_error_seconds": 2.7741747132976604,
      "end_error_seconds": 3.025380980654792,
      "interval_length_seconds": 8.870543696149703,
      "length_error_seconds": 3.243082860133761,
      "vIoU_at_03": 0.5078125,
      "vIoU_at_05": 0.234375,
      "vIoU_ge_03": 0.5078125,
      "vIoU_ge_05": 0.234375,
      "tIoU_ge_05": 0.578125,
      "tIoU_eq_0": 0.1015625,
      "boundary_MAE_seconds": 2.8997778469762263
    },
    "CleanSource": {
      "v": 0.2512425286412039,
      "t": 0.46509477679431427,
      "s": 0.5256204548915578,
      "fixed_actual_sIoU": 0.5288800918917594,
      "observed_actual_sIoU": 0.52604723375025,
      "unobserved_actual_sIoU": 0.5306732451745338,
      "boundary_MAE": 83.53515625,
      "start_error_seconds": 3.1923334330505937,
      "end_error_seconds": 3.1303359777789206,
      "interval_length_seconds": 6.262376149251184,
      "length_error_seconds": 3.3165903062722952,
      "vIoU_at_03": 0.4296875,
      "vIoU_at_05": 0.140625,
      "vIoU_ge_03": 0.4296875,
      "vIoU_ge_05": 0.140625,
      "tIoU_ge_05": 0.453125,
      "tIoU_eq_0": 0.140625,
      "boundary_MAE_seconds": 3.1613347054147574
    },
    "CleanFullMAP": {
      "v": 0.31791298341569574,
      "t": 0.5496854831561468,
      "s": 0.5496602553288483,
      "fixed_actual_sIoU": 0.5519795310346984,
      "observed_actual_sIoU": 0.5799455876867262,
      "unobserved_actual_sIoU": 0.5463700370365687,
      "boundary_MAE": 76.87109375,
      "start_error_seconds": 2.78729433478442,
      "end_error_seconds": 3.0265915296068053,
      "interval_length_seconds": 8.876093507174193,
      "length_error_seconds": 3.198348835750862,
      "vIoU_at_03": 0.5078125,
      "vIoU_at_05": 0.2265625,
      "vIoU_ge_03": 0.5078125,
      "vIoU_ge_05": 0.2265625,
      "tIoU_ge_05": 0.5859375,
      "tIoU_eq_0": 0.09375,
      "boundary_MAE_seconds": 2.9069429321956126
    }
  },
  "paired": {
    "FullMAP-Source": {
      "v": {
        "mean": 6.465497661049611,
        "CI95": [
          3.7645450438367254,
          9.130263159825155
        ],
        "queries": 128,
        "parents": 128,
        "improved": 84,
        "harmed": 30,
        "unchanged": 14,
        "declined_over5pp": 16
      },
      "t": {
        "mean": 8.057713189850261,
        "CI95": [
          4.83212887787766,
          11.265274690588255
        ],
        "queries": 128,
        "parents": 128,
        "improved": 77,
        "harmed": 36,
        "unchanged": 15,
        "declined_over5pp": 25
      },
      "observed_actual_sIoU": {
        "mean": 4.775767480009707,
        "CI95": [
          0.8927840721003848,
          8.602434339374664
        ],
        "queries": 126,
        "parents": 126,
        "improved": 83,
        "harmed": 38,
        "unchanged": 5,
        "declined_over5pp": 24
      },
      "unobserved_actual_sIoU": {
        "mean": 1.978893574626254,
        "CI95": [
          -0.8229722509405663,
          4.747604979812644
        ],
        "queries": 128,
        "parents": 128,
        "improved": 77,
        "harmed": 49,
        "unchanged": 2,
        "declined_over5pp": 25
      },
      "boundary_MAE_seconds": {
        "mean": -0.22927638682678797,
        "CI95": [
          -0.4301409832350637,
          -0.03312373971059004
        ],
        "queries": 128,
        "parents": 128,
        "improved": 70,
        "harmed": 51,
        "unchanged": 7,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.5036817403808581,
        "CI95": [
          -1.5367480797748252,
          0.36600499388785757
        ],
        "queries": 128,
        "parents": 128,
        "improved": 48,
        "harmed": 66,
        "unchanged": 14,
        "declined_over5pp": 9
      },
      "t": {
        "mean": -0.4635154549979311,
        "CI95": [
          -1.6371989187731792,
          0.4504060495428825
        ],
        "queries": 128,
        "parents": 128,
        "improved": 18,
        "harmed": 15,
        "unchanged": 95,
        "declined_over5pp": 9
      },
      "observed_actual_sIoU": {
        "mean": -0.7125917377697878,
        "CI95": [
          -2.972985193389532,
          1.4075372315803094
        ],
        "queries": 126,
        "parents": 126,
        "improved": 53,
        "harmed": 60,
        "unchanged": 13,
        "declined_over5pp": 14
      },
      "unobserved_actual_sIoU": {
        "mean": -0.022430313978939465,
        "CI95": [
          -0.8829789265364201,
          0.846633013271275
        ],
        "queries": 128,
        "parents": 128,
        "improved": 55,
        "harmed": 71,
        "unchanged": 2,
        "declined_over5pp": 11
      },
      "boundary_MAE_seconds": {
        "mean": -0.007165085219385917,
        "CI95": [
          -0.10084941292705396,
          0.08689958476039863
        ],
        "queries": 128,
        "parents": 128,
        "improved": 21,
        "harmed": 15,
        "unchanged": 92,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.3021339239812845,
        "CI95": [
          -1.0848002649161725,
          0.4683527867155226
        ],
        "queries": 128,
        "parents": 128,
        "improved": 51,
        "harmed": 57,
        "unchanged": 20,
        "declined_over5pp": 11
      },
      "t": {
        "mean": -0.06215800866493211,
        "CI95": [
          -1.4769593182422185,
          1.3138200375766513
        ],
        "queries": 128,
        "parents": 128,
        "improved": 25,
        "harmed": 25,
        "unchanged": 78,
        "declined_over5pp": 12
      },
      "observed_actual_sIoU": {
        "mean": -0.09852382413187258,
        "CI95": [
          -1.4445695107184076,
          1.402723744190569
        ],
        "queries": 126,
        "parents": 126,
        "improved": 59,
        "harmed": 59,
        "unchanged": 8,
        "declined_over5pp": 10
      },
      "unobserved_actual_sIoU": {
        "mean": -0.43164470240171593,
        "CI95": [
          -0.822179528294599,
          -0.0717426031550102
        ],
        "queries": 128,
        "parents": 128,
        "improved": 54,
        "harmed": 70,
        "unchanged": 4,
        "declined_over5pp": 6
      },
      "boundary_MAE_seconds": {
        "mean": -0.03228047161174282,
        "CI95": [
          -0.11428172237256504,
          0.04641011253738048
        ],
        "queries": 128,
        "parents": 128,
        "improved": 27,
        "harmed": 25,
        "unchanged": 76,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 25,
      "severe_v": 16,
      "severe_union_once": 34,
      "rescue_union_once": 74
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 9,
      "severe_v": 9,
      "severe_union_once": 13,
      "rescue_union_once": 13
    },
    "Source-CleanSource": {
      "severe_t": 12,
      "severe_v": 11,
      "severe_union_once": 13,
      "rescue_union_once": 16
    }
  }
}
