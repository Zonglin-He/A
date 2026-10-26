# hc2 / frame_drop_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.2354667091059674,
      "t": 0.4418222256213271,
      "s": 0.5207799715584776,
      "fixed_actual_sIoU": 0.5235683043255935,
      "observed_actual_sIoU": 0.5115607497309106,
      "unobserved_actual_sIoU": 0.526251700637182,
      "boundary_MAE": 88.25390625,
      "start_error_seconds": 3.361081950585411,
      "end_error_seconds": 3.3238380435158943,
      "interval_length_seconds": 5.549344438126726,
      "length_error_seconds": 3.40862920730872,
      "vIoU_at_03": 0.40625,
      "vIoU_at_05": 0.125,
      "vIoU_ge_03": 0.40625,
      "vIoU_ge_05": 0.125,
      "tIoU_ge_05": 0.4375,
      "tIoU_eq_0": 0.1640625,
      "boundary_MAE_seconds": 3.3424599970506526
    },
    "FullMAP": {
      "v": 0.3069294465967667,
      "t": 0.5300880557982437,
      "s": 0.5457030531900792,
      "fixed_actual_sIoU": 0.5464511425002356,
      "observed_actual_sIoU": 0.5647933397838258,
      "unobserved_actual_sIoU": 0.5427051485528797,
      "boundary_MAE": 81.80859375,
      "start_error_seconds": 3.0334306379476677,
      "end_error_seconds": 3.1534255624744265,
      "interval_length_seconds": 8.490839215214852,
      "length_error_seconds": 3.1859928850042696,
      "vIoU_at_03": 0.484375,
      "vIoU_at_05": 0.234375,
      "vIoU_ge_03": 0.484375,
      "vIoU_ge_05": 0.234375,
      "tIoU_ge_05": 0.53125,
      "tIoU_eq_0": 0.1015625,
      "boundary_MAE_seconds": 3.093428100211047
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
        "mean": 7.146273749079927,
        "CI95": [
          4.778407714450576,
          9.533042992597581
        ],
        "queries": 128,
        "parents": 128,
        "improved": 81,
        "harmed": 33,
        "unchanged": 14,
        "declined_over5pp": 16
      },
      "t": {
        "mean": 8.826583017691664,
        "CI95": [
          5.71710306064926,
          11.95469783651904
        ],
        "queries": 128,
        "parents": 128,
        "improved": 80,
        "harmed": 29,
        "unchanged": 19,
        "declined_over5pp": 21
      },
      "observed_actual_sIoU": {
        "mean": 5.323259005291538,
        "CI95": [
          1.2791326165689916,
          9.3596624108731
        ],
        "queries": 126,
        "parents": 126,
        "improved": 82,
        "harmed": 40,
        "unchanged": 4,
        "declined_over5pp": 24
      },
      "unobserved_actual_sIoU": {
        "mean": 1.6453447915697672,
        "CI95": [
          -0.9568436513855156,
          4.211296487070098
        ],
        "queries": 128,
        "parents": 128,
        "improved": 78,
        "harmed": 48,
        "unchanged": 2,
        "declined_over5pp": 24
      },
      "boundary_MAE_seconds": {
        "mean": -0.24903189683960547,
        "CI95": [
          -0.4369949089991597,
          -0.057110780868261234
        ],
        "queries": 128,
        "parents": 128,
        "improved": 79,
        "harmed": 42,
        "unchanged": 7,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -1.098353681892906,
        "CI95": [
          -2.581473573783798,
          0.3048256547491238
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 68,
        "unchanged": 14,
        "declined_over5pp": 23
      },
      "t": {
        "mean": -1.9597427357903137,
        "CI95": [
          -3.839720524607592,
          -0.1891538256505371
        ],
        "queries": 128,
        "parents": 128,
        "improved": 23,
        "harmed": 36,
        "unchanged": 69,
        "declined_over5pp": 22
      },
      "observed_actual_sIoU": {
        "mean": -1.51522479029004,
        "CI95": [
          -3.7270755941110525,
          0.6991031835903263
        ],
        "queries": 126,
        "parents": 126,
        "improved": 50,
        "harmed": 61,
        "unchanged": 15,
        "declined_over5pp": 24
      },
      "unobserved_actual_sIoU": {
        "mean": -0.36648884836890233,
        "CI95": [
          -1.6741888877783002,
          0.8924556372303519
        ],
        "queries": 128,
        "parents": 128,
        "improved": 57,
        "harmed": 69,
        "unchanged": 2,
        "declined_over5pp": 17
      },
      "boundary_MAE_seconds": {
        "mean": 0.18648516801543474,
        "CI95": [
          0.04799113561565021,
          0.33407915241492786
        ],
        "queries": 128,
        "parents": 128,
        "improved": 26,
        "harmed": 38,
        "unchanged": 64,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -1.5775819535236473,
        "CI95": [
          -2.646739855191941,
          -0.4732573914939246
        ],
        "queries": 128,
        "parents": 128,
        "improved": 40,
        "harmed": 67,
        "unchanged": 21,
        "declined_over5pp": 21
      },
      "t": {
        "mean": -2.3272551172987193,
        "CI95": [
          -3.997744621539041,
          -0.6624931175005706
        ],
        "queries": 128,
        "parents": 128,
        "improved": 18,
        "harmed": 45,
        "unchanged": 65,
        "declined_over5pp": 32
      },
      "observed_actual_sIoU": {
        "mean": -1.4486484019339543,
        "CI95": [
          -3.0316256189972073,
          0.1836408614895156
        ],
        "queries": 126,
        "parents": 126,
        "improved": 45,
        "harmed": 67,
        "unchanged": 14,
        "declined_over5pp": 21
      },
      "unobserved_actual_sIoU": {
        "mean": -0.44215445373519213,
        "CI95": [
          -0.8304672810767171,
          -0.09621495248662705
        ],
        "queries": 128,
        "parents": 128,
        "improved": 53,
        "harmed": 69,
        "unchanged": 6,
        "declined_over5pp": 3
      },
      "boundary_MAE_seconds": {
        "mean": 0.1811252916358953,
        "CI95": [
          0.04470914489678446,
          0.3196125075367063
        ],
        "queries": 128,
        "parents": 128,
        "improved": 22,
        "harmed": 50,
        "unchanged": 56,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 21,
      "severe_v": 16,
      "severe_union_once": 30,
      "rescue_union_once": 79
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 22,
      "severe_v": 23,
      "severe_union_once": 32,
      "rescue_union_once": 18
    },
    "Source-CleanSource": {
      "severe_t": 32,
      "severe_v": 21,
      "severe_union_once": 32,
      "rescue_union_once": 8
    }
  }
}
