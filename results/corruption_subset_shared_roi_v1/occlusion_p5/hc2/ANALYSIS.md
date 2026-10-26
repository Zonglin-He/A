# hc2 / occlusion_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.2424604011989342,
      "t": 0.4486131356809804,
      "s": 0.523248279633861,
      "fixed_actual_sIoU": 0.5261126992948665,
      "observed_actual_sIoU": 0.5304814764238457,
      "unobserved_actual_sIoU": 0.5260621931029403,
      "boundary_MAE": 86.328125,
      "start_error_seconds": 3.276340926653495,
      "end_error_seconds": 3.246179695709009,
      "interval_length_seconds": 5.9087871810615,
      "length_error_seconds": 3.341956431655028,
      "vIoU_at_03": 0.3984375,
      "vIoU_at_05": 0.1484375,
      "vIoU_ge_03": 0.3984375,
      "vIoU_ge_05": 0.1484375,
      "tIoU_ge_05": 0.4453125,
      "tIoU_eq_0": 0.15625,
      "boundary_MAE_seconds": 3.2612603111812524
    },
    "FullMAP": {
      "v": 0.3090057349154102,
      "t": 0.5321163817659897,
      "s": 0.5482285446728865,
      "fixed_actual_sIoU": 0.5492159248157641,
      "observed_actual_sIoU": 0.5755841521155782,
      "unobserved_actual_sIoU": 0.5434248692250393,
      "boundary_MAE": 81.18359375,
      "start_error_seconds": 3.030294904853723,
      "end_error_seconds": 3.0967936851308466,
      "interval_length_seconds": 8.63445167557369,
      "length_error_seconds": 3.2671347470164944,
      "vIoU_at_03": 0.515625,
      "vIoU_at_05": 0.234375,
      "vIoU_ge_03": 0.515625,
      "vIoU_ge_05": 0.234375,
      "tIoU_ge_05": 0.5390625,
      "tIoU_eq_0": 0.109375,
      "boundary_MAE_seconds": 3.063544294992285
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
        "mean": 6.654533371647603,
        "CI95": [
          3.9823745711026035,
          9.234035432139818
        ],
        "queries": 128,
        "parents": 128,
        "improved": 84,
        "harmed": 29,
        "unchanged": 15,
        "declined_over5pp": 18
      },
      "t": {
        "mean": 8.35032460850093,
        "CI95": [
          5.227204165902974,
          11.587161241022857
        ],
        "queries": 128,
        "parents": 128,
        "improved": 78,
        "harmed": 32,
        "unchanged": 18,
        "declined_over5pp": 19
      },
      "observed_actual_sIoU": {
        "mean": 4.5102675691732514,
        "CI95": [
          0.44966702878378756,
          8.539401339057509
        ],
        "queries": 125,
        "parents": 125,
        "improved": 81,
        "harmed": 38,
        "unchanged": 6,
        "declined_over5pp": 27
      },
      "unobserved_actual_sIoU": {
        "mean": 1.7362676122099008,
        "CI95": [
          -1.046211815338744,
          4.482881787053624
        ],
        "queries": 128,
        "parents": 128,
        "improved": 79,
        "harmed": 47,
        "unchanged": 2,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": -0.19771601618896711,
        "CI95": [
          -0.37986552468193474,
          -0.01911963828086921
        ],
        "queries": 128,
        "parents": 128,
        "improved": 82,
        "harmed": 41,
        "unchanged": 5,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.8907248500285478,
        "CI95": [
          -1.879279211694161,
          0.07185998571226304
        ],
        "queries": 128,
        "parents": 128,
        "improved": 52,
        "harmed": 62,
        "unchanged": 14,
        "declined_over5pp": 20
      },
      "t": {
        "mean": -1.7569101390157171,
        "CI95": [
          -3.177834674357563,
          -0.393148981319445
        ],
        "queries": 128,
        "parents": 128,
        "improved": 17,
        "harmed": 30,
        "unchanged": 81,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": -0.33723397839840125,
        "CI95": [
          -2.8117823196129965,
          2.0863585755034397
        ],
        "queries": 125,
        "parents": 125,
        "improved": 50,
        "harmed": 60,
        "unchanged": 15,
        "declined_over5pp": 19
      },
      "unobserved_actual_sIoU": {
        "mean": -0.29451678115293345,
        "CI95": [
          -1.3951129453657363,
          0.8056345321831084
        ],
        "queries": 128,
        "parents": 128,
        "improved": 58,
        "harmed": 68,
        "unchanged": 2,
        "declined_over5pp": 20
      },
      "boundary_MAE_seconds": {
        "mean": 0.1566013627966727,
        "CI95": [
          0.03513559492307062,
          0.28711752205712715
        ],
        "queries": 128,
        "parents": 128,
        "improved": 19,
        "harmed": 32,
        "unchanged": 77,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.878212744226966,
        "CI95": [
          -1.8503804941399509,
          0.1180236838191133
        ],
        "queries": 128,
        "parents": 128,
        "improved": 41,
        "harmed": 66,
        "unchanged": 21,
        "declined_over5pp": 17
      },
      "t": {
        "mean": -1.6481641113333882,
        "CI95": [
          -3.172776210957412,
          -0.17518411804570766
        ],
        "queries": 128,
        "parents": 128,
        "improved": 16,
        "harmed": 30,
        "unchanged": 82,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 0.5859862912904851,
        "CI95": [
          -1.1592645331918814,
          2.4127018229345096
        ],
        "queries": 125,
        "parents": 125,
        "improved": 52,
        "harmed": 64,
        "unchanged": 9,
        "declined_over5pp": 9
      },
      "unobserved_actual_sIoU": {
        "mean": -0.46110520715935666,
        "CI95": [
          -1.0009190381749418,
          0.127382840876102
        ],
        "queries": 128,
        "parents": 128,
        "improved": 59,
        "harmed": 65,
        "unchanged": 4,
        "declined_over5pp": 6
      },
      "boundary_MAE_seconds": {
        "mean": 0.09992560576649492,
        "CI95": [
          -0.020719979096865102,
          0.22360828383705297
        ],
        "queries": 128,
        "parents": 128,
        "improved": 21,
        "harmed": 31,
        "unchanged": 76,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 19,
      "severe_v": 18,
      "severe_union_once": 29,
      "rescue_union_once": 76
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 23,
      "severe_v": 20,
      "severe_union_once": 27,
      "rescue_union_once": 16
    },
    "Source-CleanSource": {
      "severe_t": 23,
      "severe_v": 17,
      "severe_union_once": 24,
      "rescue_union_once": 10
    }
  }
}
