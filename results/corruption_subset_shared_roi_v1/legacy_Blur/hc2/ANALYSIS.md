# hc2 / legacy_Blur

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.24345614688514405,
      "t": 0.44460067273236864,
      "s": 0.5258126463223869,
      "fixed_actual_sIoU": 0.5299878666959631,
      "observed_actual_sIoU": 0.5210502125121125,
      "unobserved_actual_sIoU": 0.5320789409657128,
      "boundary_MAE": 86.17578125,
      "start_error_seconds": 3.230439259469976,
      "end_error_seconds": 3.3012918223994925,
      "interval_length_seconds": 6.231536152640133,
      "length_error_seconds": 3.337438472287138,
      "vIoU_at_03": 0.4296875,
      "vIoU_at_05": 0.125,
      "vIoU_ge_03": 0.4296875,
      "vIoU_ge_05": 0.125,
      "tIoU_ge_05": 0.4765625,
      "tIoU_eq_0": 0.1328125,
      "boundary_MAE_seconds": 3.265865540934734
    },
    "FullMAP": {
      "v": 0.31167634897874263,
      "t": 0.5298571647361152,
      "s": 0.5544595572516857,
      "fixed_actual_sIoU": 0.5570615771699022,
      "observed_actual_sIoU": 0.5825382890147207,
      "unobserved_actual_sIoU": 0.5528287049454714,
      "boundary_MAE": 80.07421875,
      "start_error_seconds": 2.8431938356993953,
      "end_error_seconds": 3.1926227215724334,
      "interval_length_seconds": 8.921264745534138,
      "length_error_seconds": 3.440587404196891,
      "vIoU_at_03": 0.5078125,
      "vIoU_at_05": 0.203125,
      "vIoU_ge_03": 0.5078125,
      "vIoU_ge_05": 0.203125,
      "tIoU_ge_05": 0.546875,
      "tIoU_eq_0": 0.078125,
      "boundary_MAE_seconds": 3.017908278635914
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
        "mean": 6.822020209359855,
        "CI95": [
          4.208737080499312,
          9.397639309220112
        ],
        "queries": 128,
        "parents": 128,
        "improved": 84,
        "harmed": 33,
        "unchanged": 11,
        "declined_over5pp": 16
      },
      "t": {
        "mean": 8.525649200374655,
        "CI95": [
          5.410289549262785,
          11.639011414137931
        ],
        "queries": 128,
        "parents": 128,
        "improved": 85,
        "harmed": 28,
        "unchanged": 15,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 6.148807650260821,
        "CI95": [
          2.211430507879954,
          10.037847302458292
        ],
        "queries": 127,
        "parents": 127,
        "improved": 88,
        "harmed": 34,
        "unchanged": 5,
        "declined_over5pp": 21
      },
      "unobserved_actual_sIoU": {
        "mean": 2.0749763979758473,
        "CI95": [
          -0.5977901842898592,
          4.754813875527208
        ],
        "queries": 128,
        "parents": 128,
        "improved": 83,
        "harmed": 44,
        "unchanged": 1,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": -0.24795726229881956,
        "CI95": [
          -0.46660020414035397,
          -0.03054266790470222
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
        "mean": -0.623663443695311,
        "CI95": [
          -2.062136178933079,
          0.7123747515501835
        ],
        "queries": 128,
        "parents": 128,
        "improved": 53,
        "harmed": 64,
        "unchanged": 11,
        "declined_over5pp": 20
      },
      "t": {
        "mean": -1.982831842003167,
        "CI95": [
          -3.897437142816227,
          -0.23533164917118554
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 40,
        "unchanged": 60,
        "declined_over5pp": 25
      },
      "observed_actual_sIoU": {
        "mean": 0.13183506015430954,
        "CI95": [
          -2.548060263013463,
          2.904217699796852
        ],
        "queries": 126,
        "parents": 126,
        "improved": 59,
        "harmed": 63,
        "unchanged": 4,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 0.645866790890264,
        "CI95": [
          -1.2741805259083117,
          2.6949013698492825
        ],
        "queries": 128,
        "parents": 128,
        "improved": 68,
        "harmed": 59,
        "unchanged": 1,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": 0.1109653464403018,
        "CI95": [
          -0.04554359189198826,
          0.2686205065956307
        ],
        "queries": 128,
        "parents": 128,
        "improved": 26,
        "harmed": 46,
        "unchanged": 56,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.7786381756059816,
        "CI95": [
          -1.8079694974164566,
          0.23709113271376017
        ],
        "queries": 128,
        "parents": 128,
        "improved": 42,
        "harmed": 68,
        "unchanged": 18,
        "declined_over5pp": 12
      },
      "t": {
        "mean": -2.049410406194565,
        "CI95": [
          -3.9259420075458507,
          -0.2651318202761246
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 43,
        "unchanged": 55,
        "declined_over5pp": 30
      },
      "observed_actual_sIoU": {
        "mean": -0.5756868669946605,
        "CI95": [
          -2.0384991197980527,
          0.8441935031539188
        ],
        "queries": 126,
        "parents": 126,
        "improved": 48,
        "harmed": 69,
        "unchanged": 9,
        "declined_over5pp": 14
      },
      "unobserved_actual_sIoU": {
        "mean": 0.14056957911789403,
        "CI95": [
          -0.5215068025813011,
          0.8830320282755597
        ],
        "queries": 128,
        "parents": 128,
        "improved": 51,
        "harmed": 73,
        "unchanged": 4,
        "declined_over5pp": 5
      },
      "boundary_MAE_seconds": {
        "mean": 0.10453083551997648,
        "CI95": [
          -0.04867299211617987,
          0.26976408809615426
        ],
        "queries": 128,
        "parents": 128,
        "improved": 32,
        "harmed": 48,
        "unchanged": 48,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 23,
      "severe_v": 16,
      "severe_union_once": 32,
      "rescue_union_once": 74
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 25,
      "severe_v": 20,
      "severe_union_once": 33,
      "rescue_union_once": 21
    },
    "Source-CleanSource": {
      "severe_t": 30,
      "severe_v": 12,
      "severe_union_once": 30,
      "rescue_union_once": 17
    }
  }
}
