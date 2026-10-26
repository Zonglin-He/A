# vidstg / exposure_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.13287101676935736,
      "t": 0.32044881301450895,
      "s": 0.36034616747759024,
      "fixed_actual_sIoU": 0.3644731407851123,
      "observed_actual_sIoU": 0.3697558378338267,
      "unobserved_actual_sIoU": 0.36429469214200977,
      "boundary_MAE": 218.6484375,
      "start_error_seconds": 7.124568675296041,
      "end_error_seconds": 8.307322539514258,
      "interval_length_seconds": 7.7554300554500255,
      "length_error_seconds": 7.709101073259496,
      "vIoU_at_03": 0.1640625,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.1640625,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.3046875,
      "tIoU_eq_0": 0.2421875,
      "boundary_MAE_seconds": 7.715945607405149
    },
    "FullMAP": {
      "v": 0.15504156348713394,
      "t": 0.35371283743867266,
      "s": 0.38814718163010287,
      "fixed_actual_sIoU": 0.3920856403587264,
      "observed_actual_sIoU": 0.46129449774546394,
      "unobserved_actual_sIoU": 0.3859049693189088,
      "boundary_MAE": 206.86328125,
      "start_error_seconds": 7.281828658689089,
      "end_error_seconds": 7.381950815424364,
      "interval_length_seconds": 11.41095187371857,
      "length_error_seconds": 8.368438275080857,
      "vIoU_at_03": 0.1953125,
      "vIoU_at_05": 0.1015625,
      "vIoU_ge_03": 0.1953125,
      "vIoU_ge_05": 0.1015625,
      "tIoU_ge_05": 0.34375,
      "tIoU_eq_0": 0.1640625,
      "boundary_MAE_seconds": 7.331889737056727
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
        "mean": 2.2170546717776594,
        "CI95": [
          0.13481553068288543,
          4.346651289623619
        ],
        "queries": 128,
        "parents": 128,
        "improved": 70,
        "harmed": 33,
        "unchanged": 25,
        "declined_over5pp": 20
      },
      "t": {
        "mean": 3.326402442416361,
        "CI95": [
          1.0731711983857386,
          5.792709608347188
        ],
        "queries": 128,
        "parents": 128,
        "improved": 60,
        "harmed": 41,
        "unchanged": 27,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 9.153865991163721,
        "CI95": [
          3.812715350269891,
          14.537884372578946
        ],
        "queries": 104,
        "parents": 104,
        "improved": 69,
        "harmed": 32,
        "unchanged": 3,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 2.1610277176898998,
        "CI95": [
          -1.8883853134947315,
          6.135701344380873
        ],
        "queries": 128,
        "parents": 128,
        "improved": 76,
        "harmed": 49,
        "unchanged": 3,
        "declined_over5pp": 32
      },
      "boundary_MAE_seconds": {
        "mean": -0.3840558703484224,
        "CI95": [
          -1.0295372360600947,
          0.18315815900651422
        ],
        "queries": 128,
        "parents": 128,
        "improved": 62,
        "harmed": 56,
        "unchanged": 10,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": 0.16489730803152655,
        "CI95": [
          -0.30530117453285843,
          0.6691804872043892
        ],
        "queries": 128,
        "parents": 128,
        "improved": 34,
        "harmed": 35,
        "unchanged": 59,
        "declined_over5pp": 2
      },
      "t": {
        "mean": 0.16884901202671249,
        "CI95": [
          -0.7869211526638996,
          1.2244000554887358
        ],
        "queries": 128,
        "parents": 128,
        "improved": 18,
        "harmed": 21,
        "unchanged": 89,
        "declined_over5pp": 7
      },
      "observed_actual_sIoU": {
        "mean": 1.2204832205637908,
        "CI95": [
          0.016109362386678674,
          2.646001970767731
        ],
        "queries": 103,
        "parents": 103,
        "improved": 29,
        "harmed": 33,
        "unchanged": 41,
        "declined_over5pp": 3
      },
      "unobserved_actual_sIoU": {
        "mean": 0.3829222752060873,
        "CI95": [
          -0.38733998828323707,
          1.3636026950657734
        ],
        "queries": 128,
        "parents": 128,
        "improved": 43,
        "harmed": 40,
        "unchanged": 45,
        "declined_over5pp": 6
      },
      "boundary_MAE_seconds": {
        "mean": 0.06054019059277063,
        "CI95": [
          -0.09581414110488924,
          0.2392884693270209
        ],
        "queries": 128,
        "parents": 128,
        "improved": 22,
        "harmed": 26,
        "unchanged": 80,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.0358010484075045,
        "CI95": [
          -0.2704146191552526,
          0.20626317062013036
        ],
        "queries": 128,
        "parents": 128,
        "improved": 26,
        "harmed": 38,
        "unchanged": 64,
        "declined_over5pp": 1
      },
      "t": {
        "mean": -0.07665345360826507,
        "CI95": [
          -0.9203368730336002,
          0.6602689386511066
        ],
        "queries": 128,
        "parents": 128,
        "improved": 13,
        "harmed": 13,
        "unchanged": 102,
        "declined_over5pp": 6
      },
      "observed_actual_sIoU": {
        "mean": -0.21782867631419695,
        "CI95": [
          -0.49761464910870773,
          -0.027563018895131453
        ],
        "queries": 103,
        "parents": 103,
        "improved": 29,
        "harmed": 36,
        "unchanged": 38,
        "declined_over5pp": 1
      },
      "unobserved_actual_sIoU": {
        "mean": 0.016533031854799547,
        "CI95": [
          -0.1237721235831924,
          0.15588488155489158
        ],
        "queries": 128,
        "parents": 128,
        "improved": 38,
        "harmed": 53,
        "unchanged": 37,
        "declined_over5pp": 1
      },
      "boundary_MAE_seconds": {
        "mean": 0.026029040683302952,
        "CI95": [
          -0.05354249562052923,
          0.13447732710693364
        ],
        "queries": 128,
        "parents": 128,
        "improved": 16,
        "harmed": 18,
        "unchanged": 94,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 23,
      "severe_v": 20,
      "severe_union_once": 34,
      "rescue_union_once": 60
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 7,
      "severe_v": 2,
      "severe_union_once": 8,
      "rescue_union_once": 11
    },
    "Source-CleanSource": {
      "severe_t": 6,
      "severe_v": 1,
      "severe_union_once": 6,
      "rescue_union_once": 7
    }
  }
}
