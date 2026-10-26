# hc2 / frame_drop_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.22203590169572687,
      "t": 0.417676363813229,
      "s": 0.5159758168789974,
      "fixed_actual_sIoU": 0.5189546701418732,
      "observed_actual_sIoU": 0.5233621618461325,
      "unobserved_actual_sIoU": 0.51880405336305,
      "boundary_MAE": 92.96484375,
      "start_error_seconds": 3.5435438770624597,
      "end_error_seconds": 3.473218041784234,
      "interval_length_seconds": 5.172901887413026,
      "length_error_seconds": 3.5555278975320856,
      "vIoU_at_03": 0.3671875,
      "vIoU_at_05": 0.09375,
      "vIoU_ge_03": 0.3671875,
      "vIoU_ge_05": 0.09375,
      "tIoU_ge_05": 0.40625,
      "tIoU_eq_0": 0.1796875,
      "boundary_MAE_seconds": 3.508380959423347
    },
    "FullMAP": {
      "v": 0.30195492136124735,
      "t": 0.5224559681418299,
      "s": 0.545058164183943,
      "fixed_actual_sIoU": 0.5455173405570779,
      "observed_actual_sIoU": 0.5688774829380318,
      "unobserved_actual_sIoU": 0.5393654362473996,
      "boundary_MAE": 82.7734375,
      "start_error_seconds": 3.1374226703008294,
      "end_error_seconds": 3.1318798230462983,
      "interval_length_seconds": 8.273145905559643,
      "length_error_seconds": 3.0866912860262463,
      "vIoU_at_03": 0.46875,
      "vIoU_at_05": 0.2109375,
      "vIoU_ge_03": 0.46875,
      "vIoU_ge_05": 0.2109375,
      "tIoU_ge_05": 0.53125,
      "tIoU_eq_0": 0.1171875,
      "boundary_MAE_seconds": 3.1346512466735645
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
        "mean": 7.9919019665520485,
        "CI95": [
          5.585448670958435,
          10.334669196213241
        ],
        "queries": 128,
        "parents": 128,
        "improved": 80,
        "harmed": 31,
        "unchanged": 17,
        "declined_over5pp": 15
      },
      "t": {
        "mean": 10.477960432860094,
        "CI95": [
          7.043837741396486,
          14.000336503251658
        ],
        "queries": 128,
        "parents": 128,
        "improved": 83,
        "harmed": 27,
        "unchanged": 18,
        "declined_over5pp": 16
      },
      "observed_actual_sIoU": {
        "mean": 4.551532109189916,
        "CI95": [
          0.6200554354728505,
          8.462546203979503
        ],
        "queries": 126,
        "parents": 126,
        "improved": 84,
        "harmed": 36,
        "unchanged": 6,
        "declined_over5pp": 27
      },
      "unobserved_actual_sIoU": {
        "mean": 2.0561382884349504,
        "CI95": [
          -0.6421145632366934,
          4.767159207624559
        ],
        "queries": 128,
        "parents": 128,
        "improved": 77,
        "harmed": 49,
        "unchanged": 2,
        "declined_over5pp": 27
      },
      "boundary_MAE_seconds": {
        "mean": -0.3737297127497828,
        "CI95": [
          -0.6034766319415213,
          -0.15275679404106685
        ],
        "queries": 128,
        "parents": 128,
        "improved": 85,
        "harmed": 39,
        "unchanged": 4,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -1.595806205444834,
        "CI95": [
          -3.4567150591960596,
          0.2483181147629198
        ],
        "queries": 128,
        "parents": 128,
        "improved": 48,
        "harmed": 67,
        "unchanged": 13,
        "declined_over5pp": 32
      },
      "t": {
        "mean": -2.7229515014316967,
        "CI95": [
          -5.0275531252194705,
          -0.43778149506975517
        ],
        "queries": 128,
        "parents": 128,
        "improved": 26,
        "harmed": 44,
        "unchanged": 58,
        "declined_over5pp": 34
      },
      "observed_actual_sIoU": {
        "mean": -1.1068104748694665,
        "CI95": [
          -4.248711605747076,
          2.010981608758203
        ],
        "queries": 126,
        "parents": 126,
        "improved": 58,
        "harmed": 56,
        "unchanged": 12,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": -0.7004600789169039,
        "CI95": [
          -2.680475501227262,
          1.1062274119622029
        ],
        "queries": 128,
        "parents": 128,
        "improved": 60,
        "harmed": 66,
        "unchanged": 2,
        "declined_over5pp": 23
      },
      "boundary_MAE_seconds": {
        "mean": 0.2277083144779517,
        "CI95": [
          0.04784310515143146,
          0.4176151327132635
        ],
        "queries": 128,
        "parents": 128,
        "improved": 32,
        "harmed": 46,
        "unchanged": 50,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -2.920662694547698,
        "CI95": [
          -4.520938221649477,
          -1.3531657674414022
        ],
        "queries": 128,
        "parents": 128,
        "improved": 42,
        "harmed": 66,
        "unchanged": 20,
        "declined_over5pp": 30
      },
      "t": {
        "mean": -4.7418412981085325,
        "CI95": [
          -7.650911915601839,
          -2.0032980478844546
        ],
        "queries": 128,
        "parents": 128,
        "improved": 21,
        "harmed": 53,
        "unchanged": 54,
        "declined_over5pp": 41
      },
      "observed_actual_sIoU": {
        "mean": -0.26850719041176074,
        "CI95": [
          -2.122444029335998,
          1.6335031140164211
        ],
        "queries": 126,
        "parents": 126,
        "improved": 64,
        "harmed": 54,
        "unchanged": 8,
        "declined_over5pp": 21
      },
      "unobserved_actual_sIoU": {
        "mean": -1.1869191811483775,
        "CI95": [
          -1.700832567577058,
          -0.7162854871904257
        ],
        "queries": 128,
        "parents": 128,
        "improved": 44,
        "harmed": 80,
        "unchanged": 4,
        "declined_over5pp": 11
      },
      "boundary_MAE_seconds": {
        "mean": 0.34704625400858957,
        "CI95": [
          0.13081166867508606,
          0.5849464474827839
        ],
        "queries": 128,
        "parents": 128,
        "improved": 26,
        "harmed": 56,
        "unchanged": 46,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 16,
      "severe_v": 15,
      "severe_union_once": 23,
      "rescue_union_once": 79
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 34,
      "severe_v": 32,
      "severe_union_once": 41,
      "rescue_union_once": 24
    },
    "Source-CleanSource": {
      "severe_t": 41,
      "severe_v": 30,
      "severe_union_once": 44,
      "rescue_union_once": 11
    }
  }
}
