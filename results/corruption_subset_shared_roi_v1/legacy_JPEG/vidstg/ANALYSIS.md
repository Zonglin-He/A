# vidstg / legacy_JPEG

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.13369801906485787,
      "t": 0.31908496782181994,
      "s": 0.3583799825256302,
      "fixed_actual_sIoU": 0.36294632344186006,
      "observed_actual_sIoU": 0.37293520976835903,
      "unobserved_actual_sIoU": 0.3624832063056871,
      "boundary_MAE": 213.30859375,
      "start_error_seconds": 6.76638958656452,
      "end_error_seconds": 8.28824670143607,
      "interval_length_seconds": 7.6803153530307675,
      "length_error_seconds": 7.4339212688903835,
      "vIoU_at_03": 0.1640625,
      "vIoU_at_05": 0.046875,
      "vIoU_ge_03": 0.1640625,
      "vIoU_ge_05": 0.046875,
      "tIoU_ge_05": 0.265625,
      "tIoU_eq_0": 0.2265625,
      "boundary_MAE_seconds": 7.5273181440002945
    },
    "FullMAP": {
      "v": 0.14595390636581723,
      "t": 0.351348981855807,
      "s": 0.3650623260999042,
      "fixed_actual_sIoU": 0.3715672070714772,
      "observed_actual_sIoU": 0.4401802597546084,
      "unobserved_actual_sIoU": 0.36457655330900063,
      "boundary_MAE": 205.09375,
      "start_error_seconds": 7.165338165042597,
      "end_error_seconds": 7.304409428398792,
      "interval_length_seconds": 11.42281924942972,
      "length_error_seconds": 8.246735928855475,
      "vIoU_at_03": 0.171875,
      "vIoU_at_05": 0.1171875,
      "vIoU_ge_03": 0.171875,
      "vIoU_ge_05": 0.1171875,
      "tIoU_ge_05": 0.3203125,
      "tIoU_eq_0": 0.171875,
      "boundary_MAE_seconds": 7.234873796720694
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
        "mean": 1.2255887300959376,
        "CI95": [
          -0.8329424530362921,
          3.3649143241200443
        ],
        "queries": 128,
        "parents": 128,
        "improved": 59,
        "harmed": 41,
        "unchanged": 28,
        "declined_over5pp": 23
      },
      "t": {
        "mean": 3.226401403398703,
        "CI95": [
          0.9481504985633689,
          5.749546798973092
        ],
        "queries": 128,
        "parents": 128,
        "improved": 63,
        "harmed": 39,
        "unchanged": 26,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 6.724504998624925,
        "CI95": [
          1.7843461204717768,
          11.775454606425221
        ],
        "queries": 106,
        "parents": 106,
        "improved": 66,
        "harmed": 35,
        "unchanged": 5,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 0.2093347003313546,
        "CI95": [
          -3.675084430570082,
          4.076192629153513
        ],
        "queries": 128,
        "parents": 128,
        "improved": 66,
        "harmed": 59,
        "unchanged": 3,
        "declined_over5pp": 34
      },
      "boundary_MAE_seconds": {
        "mean": -0.2924443472796003,
        "CI95": [
          -0.9292647245427251,
          0.297771065133561
        ],
        "queries": 128,
        "parents": 128,
        "improved": 66,
        "harmed": 50,
        "unchanged": 12,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.7438684041001439,
        "CI95": [
          -1.5502265052919615,
          -0.00943433204518863
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 54,
        "unchanged": 28,
        "declined_over5pp": 9
      },
      "t": {
        "mean": -0.06753654625985156,
        "CI95": [
          -1.6422391871160575,
          1.3769394056952273
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 42,
        "unchanged": 56,
        "declined_over5pp": 13
      },
      "observed_actual_sIoU": {
        "mean": -1.7393369978980429,
        "CI95": [
          -4.36501251583123,
          0.676423695949774
        ],
        "queries": 102,
        "parents": 102,
        "improved": 42,
        "harmed": 51,
        "unchanged": 9,
        "declined_over5pp": 19
      },
      "unobserved_actual_sIoU": {
        "mean": -1.7499193257847234,
        "CI95": [
          -3.70524806713685,
          0.08082512566985067
        ],
        "queries": 128,
        "parents": 128,
        "improved": 54,
        "harmed": 59,
        "unchanged": 15,
        "declined_over5pp": 20
      },
      "boundary_MAE_seconds": {
        "mean": -0.036475749743261746,
        "CI95": [
          -0.3756908510393434,
          0.2833416537542177
        ],
        "queries": 128,
        "parents": 128,
        "improved": 37,
        "harmed": 49,
        "unchanged": 42,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": 0.0468991811425464,
        "CI95": [
          -0.4740310778668671,
          0.6000340082430166
        ],
        "queries": 128,
        "parents": 128,
        "improved": 42,
        "harmed": 53,
        "unchanged": 33,
        "declined_over5pp": 6
      },
      "t": {
        "mean": -0.2130379728771708,
        "CI95": [
          -1.6575978823948294,
          1.1710461164879882
        ],
        "queries": 128,
        "parents": 128,
        "improved": 29,
        "harmed": 35,
        "unchanged": 64,
        "declined_over5pp": 10
      },
      "observed_actual_sIoU": {
        "mean": -0.13927010957438077,
        "CI95": [
          -1.1512695943203308,
          0.8284420975432427
        ],
        "queries": 102,
        "parents": 102,
        "improved": 50,
        "harmed": 42,
        "unchanged": 10,
        "declined_over5pp": 7
      },
      "unobserved_actual_sIoU": {
        "mean": -0.16461555177746648,
        "CI95": [
          -0.7815547935541096,
          0.41240914548357227
        ],
        "queries": 128,
        "parents": 128,
        "improved": 55,
        "harmed": 67,
        "unchanged": 6,
        "declined_over5pp": 6
      },
      "boundary_MAE_seconds": {
        "mean": -0.16259842272155162,
        "CI95": [
          -0.36700090251004874,
          0.022160570171395253
        ],
        "queries": 128,
        "parents": 128,
        "improved": 37,
        "harmed": 44,
        "unchanged": 47,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 23,
      "severe_v": 23,
      "severe_union_once": 35,
      "rescue_union_once": 57
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 13,
      "severe_v": 9,
      "severe_union_once": 20,
      "rescue_union_once": 23
    },
    "Source-CleanSource": {
      "severe_t": 10,
      "severe_v": 6,
      "severe_union_once": 13,
      "rescue_union_once": 12
    }
  }
}
