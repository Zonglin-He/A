# vidstg / motion_blur_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.13369560899613897,
      "t": 0.31617554599225095,
      "s": 0.35911270159995806,
      "fixed_actual_sIoU": 0.36331077839639875,
      "observed_actual_sIoU": 0.3638093577190989,
      "unobserved_actual_sIoU": 0.36337935814710354,
      "boundary_MAE": 219.56640625,
      "start_error_seconds": 7.185981181444561,
      "end_error_seconds": 8.282291714543922,
      "interval_length_seconds": 7.593641724647147,
      "length_error_seconds": 7.85144113412954,
      "vIoU_at_03": 0.171875,
      "vIoU_at_05": 0.046875,
      "vIoU_ge_03": 0.171875,
      "vIoU_ge_05": 0.046875,
      "tIoU_ge_05": 0.28125,
      "tIoU_eq_0": 0.234375,
      "boundary_MAE_seconds": 7.734136447994241
    },
    "FullMAP": {
      "v": 0.1477983890437993,
      "t": 0.3393508003528488,
      "s": 0.37955251330675155,
      "fixed_actual_sIoU": 0.3851764406335575,
      "observed_actual_sIoU": 0.4360807730265061,
      "unobserved_actual_sIoU": 0.38041659660519933,
      "boundary_MAE": 204.0390625,
      "start_error_seconds": 7.234879295251105,
      "end_error_seconds": 7.182432030122024,
      "interval_length_seconds": 10.730528104581241,
      "length_error_seconds": 8.123015107824454,
      "vIoU_at_03": 0.1796875,
      "vIoU_at_05": 0.09375,
      "vIoU_ge_03": 0.1796875,
      "vIoU_ge_05": 0.09375,
      "tIoU_ge_05": 0.3203125,
      "tIoU_eq_0": 0.1796875,
      "boundary_MAE_seconds": 7.208655662686564
    },
    "CleanSource": {
      "v": 0.13322902725343239,
      "t": 0.32121534755059167,
      "s": 0.3605295478212531,
      "fixed_actual_sIoU": 0.3646913925512969,
      "observed_actual_sIoU": 0.3739578514313604,
      "unobserved_actual_sIoU": 0.3641293618234618,
      "boundary_MAE": 218.015625,
      "start_error_seconds": 7.0140726018788735,
      "end_error_seconds": 8.36576053156482,
      "interval_length_seconds": 7.829036650087549,
      "length_error_seconds": 7.678825410639762,
      "vIoU_at_03": 0.1640625,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.1640625,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.296875,
      "tIoU_eq_0": 0.2265625,
      "boundary_MAE_seconds": 7.689916566721846
    },
    "CleanFullMAP": {
      "v": 0.15339259040681869,
      "t": 0.3520243473184055,
      "s": 0.3839284008532412,
      "fixed_actual_sIoU": 0.3881660693441731,
      "observed_actual_sIoU": 0.45197783821206805,
      "unobserved_actual_sIoU": 0.38207574656684784,
      "boundary_MAE": 205.5625,
      "start_error_seconds": 7.117694327111833,
      "end_error_seconds": 7.42500476581608,
      "interval_length_seconds": 11.430284426520874,
      "length_error_seconds": 8.085201116805726,
      "vIoU_at_03": 0.1953125,
      "vIoU_at_05": 0.1015625,
      "vIoU_ge_03": 0.1953125,
      "vIoU_ge_05": 0.1015625,
      "tIoU_ge_05": 0.3359375,
      "tIoU_eq_0": 0.171875,
      "boundary_MAE_seconds": 7.271349546463956
    }
  },
  "paired": {
    "FullMAP-Source": {
      "v": {
        "mean": 1.4102780047660306,
        "CI95": [
          -0.5880042263058212,
          3.440290709229638
        ],
        "queries": 128,
        "parents": 128,
        "improved": 64,
        "harmed": 39,
        "unchanged": 25,
        "declined_over5pp": 21
      },
      "t": {
        "mean": 2.3175254360597837,
        "CI95": [
          0.38217235013802714,
          4.440607934085881
        ],
        "queries": 128,
        "parents": 128,
        "improved": 58,
        "harmed": 45,
        "unchanged": 25,
        "declined_over5pp": 25
      },
      "observed_actual_sIoU": {
        "mean": 7.22714153074072,
        "CI95": [
          2.092358959465506,
          12.42447780103934
        ],
        "queries": 104,
        "parents": 104,
        "improved": 65,
        "harmed": 34,
        "unchanged": 5,
        "declined_over5pp": 24
      },
      "unobserved_actual_sIoU": {
        "mean": 1.7037238458095794,
        "CI95": [
          -2.2734530558139503,
          5.549672737533613
        ],
        "queries": 128,
        "parents": 128,
        "improved": 69,
        "harmed": 55,
        "unchanged": 4,
        "declined_over5pp": 33
      },
      "boundary_MAE_seconds": {
        "mean": -0.5254807853076775,
        "CI95": [
          -1.1999337446482399,
          0.04336419588855174
        ],
        "queries": 128,
        "parents": 128,
        "improved": 59,
        "harmed": 61,
        "unchanged": 8,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.5594201363019405,
        "CI95": [
          -1.556124427575859,
          0.391484217080951
        ],
        "queries": 128,
        "parents": 128,
        "improved": 35,
        "harmed": 40,
        "unchanged": 53,
        "declined_over5pp": 15
      },
      "t": {
        "mean": -1.267354696555668,
        "CI95": [
          -2.769118915449388,
          0.1634112708732077
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 35,
        "unchanged": 65,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": -1.0157589246401577,
        "CI95": [
          -2.765504209561859,
          0.6531252499410388
        ],
        "queries": 100,
        "parents": 100,
        "improved": 25,
        "harmed": 39,
        "unchanged": 36,
        "declined_over5pp": 16
      },
      "unobserved_actual_sIoU": {
        "mean": -0.16591499616485458,
        "CI95": [
          -1.7015101915591613,
          1.4728481842002
        ],
        "queries": 128,
        "parents": 128,
        "improved": 42,
        "harmed": 45,
        "unchanged": 41,
        "declined_over5pp": 18
      },
      "boundary_MAE_seconds": {
        "mean": -0.06269388377739228,
        "CI95": [
          -0.4451597478704018,
          0.3094669301345556
        ],
        "queries": 128,
        "parents": 128,
        "improved": 36,
        "harmed": 40,
        "unchanged": 52,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": 0.04665817427065705,
        "CI95": [
          -0.8593250571595978,
          0.9830150419426572
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 42,
        "unchanged": 58,
        "declined_over5pp": 5
      },
      "t": {
        "mean": -0.5039801558340685,
        "CI95": [
          -2.2299911204978176,
          1.1869642428608285
        ],
        "queries": 128,
        "parents": 128,
        "improved": 23,
        "harmed": 34,
        "unchanged": 71,
        "declined_over5pp": 15
      },
      "observed_actual_sIoU": {
        "mean": -0.7330345583486553,
        "CI95": [
          -2.2719358433037526,
          0.6738792803335157
        ],
        "queries": 100,
        "parents": 100,
        "improved": 26,
        "harmed": 40,
        "unchanged": 34,
        "declined_over5pp": 10
      },
      "unobserved_actual_sIoU": {
        "mean": -0.07500036763582231,
        "CI95": [
          -0.8444727332892377,
          0.6802503050926791
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 48,
        "unchanged": 34,
        "declined_over5pp": 5
      },
      "boundary_MAE_seconds": {
        "mean": 0.04421988127239497,
        "CI95": [
          -0.330374095665606,
          0.46873796308271143
        ],
        "queries": 128,
        "parents": 128,
        "improved": 36,
        "harmed": 36,
        "unchanged": 56,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 25,
      "severe_v": 21,
      "severe_union_once": 39,
      "rescue_union_once": 52
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 20,
      "severe_v": 15,
      "severe_union_once": 23,
      "rescue_union_once": 16
    },
    "Source-CleanSource": {
      "severe_t": 15,
      "severe_v": 5,
      "severe_union_once": 16,
      "rescue_union_once": 13
    }
  }
}
