# hc2 / motion_blur_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.24236181079390354,
      "t": 0.45029759410730685,
      "s": 0.5237195194956117,
      "fixed_actual_sIoU": 0.5274467714459313,
      "observed_actual_sIoU": 0.5309399043050802,
      "unobserved_actual_sIoU": 0.5281524032791797,
      "boundary_MAE": 86.3671875,
      "start_error_seconds": 3.3316048096193764,
      "end_error_seconds": 3.20357186380381,
      "interval_length_seconds": 5.964686800165176,
      "length_error_seconds": 3.2985041899468195,
      "vIoU_at_03": 0.421875,
      "vIoU_at_05": 0.1328125,
      "vIoU_ge_03": 0.421875,
      "vIoU_ge_05": 0.1328125,
      "tIoU_ge_05": 0.4296875,
      "tIoU_eq_0": 0.1484375,
      "boundary_MAE_seconds": 3.2675883367115928
    },
    "FullMAP": {
      "v": 0.3073615762659435,
      "t": 0.5282500718529254,
      "s": 0.5435498407502322,
      "fixed_actual_sIoU": 0.5440213854135951,
      "observed_actual_sIoU": 0.559519961389857,
      "unobserved_actual_sIoU": 0.5411661409825108,
      "boundary_MAE": 82.08203125,
      "start_error_seconds": 3.0617115117525153,
      "end_error_seconds": 3.1478865250593504,
      "interval_length_seconds": 8.481471377603423,
      "length_error_seconds": 3.1429434281019795,
      "vIoU_at_03": 0.5078125,
      "vIoU_at_05": 0.21875,
      "vIoU_ge_03": 0.5078125,
      "vIoU_ge_05": 0.21875,
      "tIoU_ge_05": 0.5390625,
      "tIoU_eq_0": 0.109375,
      "boundary_MAE_seconds": 3.1047990184059335
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
        "mean": 6.499976547203998,
        "CI95": [
          3.9940888888382666,
          8.993254939021211
        ],
        "queries": 128,
        "parents": 128,
        "improved": 81,
        "harmed": 32,
        "unchanged": 15,
        "declined_over5pp": 15
      },
      "t": {
        "mean": 7.7952477745618545,
        "CI95": [
          4.615502173857368,
          11.134941188947982
        ],
        "queries": 128,
        "parents": 128,
        "improved": 77,
        "harmed": 34,
        "unchanged": 17,
        "declined_over5pp": 22
      },
      "observed_actual_sIoU": {
        "mean": 2.858005708477654,
        "CI95": [
          -0.8543448545208071,
          6.5313401719319275
        ],
        "queries": 126,
        "parents": 126,
        "improved": 79,
        "harmed": 42,
        "unchanged": 5,
        "declined_over5pp": 30
      },
      "unobserved_actual_sIoU": {
        "mean": 1.3013737703331103,
        "CI95": [
          -1.2345096556536077,
          3.7538927029437876
        ],
        "queries": 128,
        "parents": 128,
        "improved": 79,
        "harmed": 47,
        "unchanged": 2,
        "declined_over5pp": 23
      },
      "boundary_MAE_seconds": {
        "mean": -0.16278931830566,
        "CI95": [
          -0.3598121482806121,
          0.02730986872679812
        ],
        "queries": 128,
        "parents": 128,
        "improved": 76,
        "harmed": 46,
        "unchanged": 6,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -1.0551407149752192,
        "CI95": [
          -2.742632970539803,
          0.6292715959134212
        ],
        "queries": 128,
        "parents": 128,
        "improved": 50,
        "harmed": 65,
        "unchanged": 13,
        "declined_over5pp": 24
      },
      "t": {
        "mean": -2.14354113032215,
        "CI95": [
          -4.114767896137706,
          -0.1542816344146461
        ],
        "queries": 128,
        "parents": 128,
        "improved": 22,
        "harmed": 34,
        "unchanged": 72,
        "declined_over5pp": 24
      },
      "observed_actual_sIoU": {
        "mean": -2.042562629686946,
        "CI95": [
          -5.463969352469279,
          1.1623773242908488
        ],
        "queries": 126,
        "parents": 126,
        "improved": 54,
        "harmed": 61,
        "unchanged": 11,
        "declined_over5pp": 22
      },
      "unobserved_actual_sIoU": {
        "mean": -0.5203896054057848,
        "CI95": [
          -2.3409076920094445,
          1.1716228341139139
        ],
        "queries": 128,
        "parents": 128,
        "improved": 58,
        "harmed": 68,
        "unchanged": 2,
        "declined_over5pp": 21
      },
      "boundary_MAE_seconds": {
        "mean": 0.19785608621032064,
        "CI95": [
          0.04255579373244596,
          0.3573500521345504
        ],
        "queries": 128,
        "parents": 128,
        "improved": 25,
        "harmed": 38,
        "unchanged": 65,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.8880717847300335,
        "CI95": [
          -2.0277990281660943,
          0.28215770939359724
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 62,
        "unchanged": 20,
        "declined_over5pp": 17
      },
      "t": {
        "mean": -1.4797182687007462,
        "CI95": [
          -3.1759098621618067,
          0.23338509006437108
        ],
        "queries": 128,
        "parents": 128,
        "improved": 21,
        "harmed": 40,
        "unchanged": 67,
        "declined_over5pp": 24
      },
      "observed_actual_sIoU": {
        "mean": 0.489267055483022,
        "CI95": [
          -1.1995785297843864,
          2.2587740699332026
        ],
        "queries": 126,
        "parents": 126,
        "improved": 60,
        "harmed": 58,
        "unchanged": 8,
        "declined_over5pp": 13
      },
      "unobserved_actual_sIoU": {
        "mean": -0.25208418953541745,
        "CI95": [
          -0.8089269506002505,
          0.48474651424767495
        ],
        "queries": 128,
        "parents": 128,
        "improved": 55,
        "harmed": 70,
        "unchanged": 3,
        "declined_over5pp": 3
      },
      "boundary_MAE_seconds": {
        "mean": 0.10625363129683582,
        "CI95": [
          -0.01648422880633,
          0.2306901505380717
        ],
        "queries": 128,
        "parents": 128,
        "improved": 23,
        "harmed": 45,
        "unchanged": 60,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 22,
      "severe_v": 15,
      "severe_union_once": 30,
      "rescue_union_once": 75
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 24,
      "severe_v": 24,
      "severe_union_once": 30,
      "rescue_union_once": 18
    },
    "Source-CleanSource": {
      "severe_t": 24,
      "severe_v": 17,
      "severe_union_once": 25,
      "rescue_union_once": 15
    }
  }
}
