# vidstg / occlusion_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.13336073586041408,
      "t": 0.32542117205729115,
      "s": 0.3570583179956191,
      "fixed_actual_sIoU": 0.36154745349536854,
      "observed_actual_sIoU": 0.3631961359971383,
      "unobserved_actual_sIoU": 0.36071789281668043,
      "boundary_MAE": 212.83984375,
      "start_error_seconds": 6.884847336770015,
      "end_error_seconds": 8.130470578850636,
      "interval_length_seconds": 8.079370009357223,
      "length_error_seconds": 7.752622832409437,
      "vIoU_at_03": 0.1640625,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.1640625,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.3046875,
      "tIoU_eq_0": 0.2265625,
      "boundary_MAE_seconds": 7.507658957810325
    },
    "FullMAP": {
      "v": 0.1530484756607744,
      "t": 0.3570402470854769,
      "s": 0.3785927814641294,
      "fixed_actual_sIoU": 0.38271557582069987,
      "observed_actual_sIoU": 0.4405932345448308,
      "unobserved_actual_sIoU": 0.37678349376907067,
      "boundary_MAE": 204.83203125,
      "start_error_seconds": 7.082558654157296,
      "end_error_seconds": 7.432136039906659,
      "interval_length_seconds": 11.31187832981912,
      "length_error_seconds": 8.04314209995643,
      "vIoU_at_03": 0.1953125,
      "vIoU_at_05": 0.109375,
      "vIoU_ge_03": 0.1953125,
      "vIoU_ge_05": 0.109375,
      "tIoU_ge_05": 0.34375,
      "tIoU_eq_0": 0.171875,
      "boundary_MAE_seconds": 7.2573473470319785
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
        "mean": 1.9687739800360322,
        "CI95": [
          -0.13408505034664628,
          4.092426387564844
        ],
        "queries": 128,
        "parents": 128,
        "improved": 67,
        "harmed": 36,
        "unchanged": 25,
        "declined_over5pp": 21
      },
      "t": {
        "mean": 3.1619075028185737,
        "CI95": [
          0.8878655954806548,
          5.67382941600505
        ],
        "queries": 128,
        "parents": 128,
        "improved": 60,
        "harmed": 39,
        "unchanged": 29,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 7.73970985476925,
        "CI95": [
          2.2673233482837745,
          13.285275869794116
        ],
        "queries": 101,
        "parents": 101,
        "improved": 66,
        "harmed": 32,
        "unchanged": 3,
        "declined_over5pp": 26
      },
      "unobserved_actual_sIoU": {
        "mean": 1.6065600952390202,
        "CI95": [
          -2.473714503620112,
          5.572074437669928
        ],
        "queries": 128,
        "parents": 128,
        "improved": 72,
        "harmed": 53,
        "unchanged": 3,
        "declined_over5pp": 34
      },
      "boundary_MAE_seconds": {
        "mean": -0.25031161077834696,
        "CI95": [
          -0.8590420180027617,
          0.2805850013730324
        ],
        "queries": 128,
        "parents": 128,
        "improved": 61,
        "harmed": 58,
        "unchanged": 9,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.03441147460443039,
        "CI95": [
          -0.46628493309667535,
          0.444221756850182
        ],
        "queries": 128,
        "parents": 128,
        "improved": 31,
        "harmed": 38,
        "unchanged": 59,
        "declined_over5pp": 4
      },
      "t": {
        "mean": 0.5015899767071408,
        "CI95": [
          -0.43379074922863253,
          1.52730317599325
        ],
        "queries": 128,
        "parents": 128,
        "improved": 23,
        "harmed": 20,
        "unchanged": 85,
        "declined_over5pp": 6
      },
      "observed_actual_sIoU": {
        "mean": -0.2913889836594369,
        "CI95": [
          -0.929218654799826,
          0.308365850224237
        ],
        "queries": 101,
        "parents": 101,
        "improved": 23,
        "harmed": 34,
        "unchanged": 44,
        "declined_over5pp": 5
      },
      "unobserved_actual_sIoU": {
        "mean": -0.5292252797777217,
        "CI95": [
          -1.0013482691616047,
          -0.09264636604657446
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 52,
        "unchanged": 46,
        "declined_over5pp": 8
      },
      "boundary_MAE_seconds": {
        "mean": -0.014002199431978241,
        "CI95": [
          -0.1459024769207787,
          0.12726902499518625
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 26,
        "unchanged": 74,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": 0.013170860698165327,
        "CI95": [
          -0.285176574654797,
          0.3194781033863868
        ],
        "queries": 128,
        "parents": 128,
        "improved": 34,
        "harmed": 31,
        "unchanged": 63,
        "declined_over5pp": 1
      },
      "t": {
        "mean": 0.42058245066995104,
        "CI95": [
          -0.37838417996806284,
          1.5560691861117588
        ],
        "queries": 128,
        "parents": 128,
        "improved": 14,
        "harmed": 14,
        "unchanged": 100,
        "declined_over5pp": 4
      },
      "observed_actual_sIoU": {
        "mean": -0.49014033314913275,
        "CI95": [
          -1.3052897657314815,
          0.01552181824501484
        ],
        "queries": 101,
        "parents": 101,
        "improved": 31,
        "harmed": 31,
        "unchanged": 39,
        "declined_over5pp": 2
      },
      "unobserved_actual_sIoU": {
        "mean": -0.34114690067812997,
        "CI95": [
          -0.7382671874506883,
          -0.027640051449880118
        ],
        "queries": 128,
        "parents": 128,
        "improved": 52,
        "harmed": 39,
        "unchanged": 37,
        "declined_over5pp": 3
      },
      "boundary_MAE_seconds": {
        "mean": -0.1822576089115214,
        "CI95": [
          -0.4411091779475182,
          0.018894788926282904
        ],
        "queries": 128,
        "parents": 128,
        "improved": 21,
        "harmed": 19,
        "unchanged": 88,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 23,
      "severe_v": 21,
      "severe_union_once": 35,
      "rescue_union_once": 61
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 6,
      "severe_v": 4,
      "severe_union_once": 8,
      "rescue_union_once": 6
    },
    "Source-CleanSource": {
      "severe_t": 4,
      "severe_v": 1,
      "severe_union_once": 5,
      "rescue_union_once": 6
    }
  }
}
