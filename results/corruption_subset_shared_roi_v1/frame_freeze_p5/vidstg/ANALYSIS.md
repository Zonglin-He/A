# vidstg / frame_freeze_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.13026619796722178,
      "t": 0.31852225508946685,
      "s": 0.35890097480045213,
      "fixed_actual_sIoU": 0.3628923614925105,
      "observed_actual_sIoU": 0.36509643177994466,
      "unobserved_actual_sIoU": 0.36308275327923767,
      "boundary_MAE": 218.890625,
      "start_error_seconds": 7.03256084656255,
      "end_error_seconds": 8.415358989026602,
      "interval_length_seconds": 7.829374554188961,
      "length_error_seconds": 7.904882558621683,
      "vIoU_at_03": 0.1640625,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.1640625,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.28125,
      "tIoU_eq_0": 0.21875,
      "boundary_MAE_seconds": 7.723959917794575
    },
    "FullMAP": {
      "v": 0.1493927025577768,
      "t": 0.3464704191685941,
      "s": 0.38647763666642765,
      "fixed_actual_sIoU": 0.3913970015556873,
      "observed_actual_sIoU": 0.4478285030550009,
      "unobserved_actual_sIoU": 0.38587953978054496,
      "boundary_MAE": 204.09765625,
      "start_error_seconds": 7.288980205763922,
      "end_error_seconds": 7.14557396528899,
      "interval_length_seconds": 11.281606682761488,
      "length_error_seconds": 7.9405646584723915,
      "vIoU_at_03": 0.1953125,
      "vIoU_at_05": 0.0859375,
      "vIoU_ge_03": 0.1953125,
      "vIoU_ge_05": 0.0859375,
      "tIoU_ge_05": 0.328125,
      "tIoU_eq_0": 0.1875,
      "boundary_MAE_seconds": 7.217277085526456
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
        "mean": 1.912650459055504,
        "CI95": [
          -0.18340388346866668,
          4.071197735857539
        ],
        "queries": 128,
        "parents": 128,
        "improved": 64,
        "harmed": 38,
        "unchanged": 26,
        "declined_over5pp": 21
      },
      "t": {
        "mean": 2.7948164079127267,
        "CI95": [
          0.4839066430527843,
          5.436245336760729
        ],
        "queries": 128,
        "parents": 128,
        "improved": 50,
        "harmed": 48,
        "unchanged": 30,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 8.27320712750561,
        "CI95": [
          2.86641099279972,
          13.774314968636258
        ],
        "queries": 103,
        "parents": 103,
        "improved": 68,
        "harmed": 31,
        "unchanged": 4,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 2.2796786501307222,
        "CI95": [
          -1.7422369156498732,
          6.2361881521321845
        ],
        "queries": 128,
        "parents": 128,
        "improved": 74,
        "harmed": 50,
        "unchanged": 4,
        "declined_over5pp": 32
      },
      "boundary_MAE_seconds": {
        "mean": -0.5066828322681197,
        "CI95": [
          -1.2008308324120645,
          0.08696771627998842
        ],
        "queries": 128,
        "parents": 128,
        "improved": 57,
        "harmed": 59,
        "unchanged": 12,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.39998878490418693,
        "CI95": [
          -1.593016139281624,
          0.5828430223941463
        ],
        "queries": 128,
        "parents": 128,
        "improved": 33,
        "harmed": 34,
        "unchanged": 61,
        "declined_over5pp": 6
      },
      "t": {
        "mean": -0.5553928149811393,
        "CI95": [
          -1.7800944828872742,
          0.4119537650078312
        ],
        "queries": 128,
        "parents": 128,
        "improved": 15,
        "harmed": 20,
        "unchanged": 93,
        "declined_over5pp": 8
      },
      "observed_actual_sIoU": {
        "mean": 0.48462635231453866,
        "CI95": [
          -0.5319730985772635,
          1.6937687222640714
        ],
        "queries": 102,
        "parents": 102,
        "improved": 28,
        "harmed": 35,
        "unchanged": 39,
        "declined_over5pp": 3
      },
      "unobserved_actual_sIoU": {
        "mean": 0.3803793213697034,
        "CI95": [
          -0.8406052743337464,
          1.6682350264286996
        ],
        "queries": 128,
        "parents": 128,
        "improved": 40,
        "harmed": 43,
        "unchanged": 45,
        "declined_over5pp": 4
      },
      "boundary_MAE_seconds": {
        "mean": -0.05407246093749995,
        "CI95": [
          -0.29443508803481205,
          0.14867894193021144
        ],
        "queries": 128,
        "parents": 128,
        "improved": 22,
        "harmed": 22,
        "unchanged": 84,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.29628292862106337,
        "CI95": [
          -1.1264104323349624,
          0.260453776487326
        ],
        "queries": 128,
        "parents": 128,
        "improved": 33,
        "harmed": 33,
        "unchanged": 62,
        "declined_over5pp": 3
      },
      "t": {
        "mean": -0.2693092461124824,
        "CI95": [
          -1.4280972502325953,
          0.6684929450340753
        ],
        "queries": 128,
        "parents": 128,
        "improved": 17,
        "harmed": 9,
        "unchanged": 102,
        "declined_over5pp": 4
      },
      "observed_actual_sIoU": {
        "mean": -0.7786799055906861,
        "CI95": [
          -1.6965462692407989,
          -0.1876651224875843
        ],
        "queries": 102,
        "parents": 102,
        "improved": 28,
        "harmed": 37,
        "unchanged": 37,
        "declined_over5pp": 3
      },
      "unobserved_actual_sIoU": {
        "mean": -0.10466085442240657,
        "CI95": [
          -0.5429245721936587,
          0.3162638310788954
        ],
        "queries": 128,
        "parents": 128,
        "improved": 48,
        "harmed": 44,
        "unchanged": 36,
        "declined_over5pp": 3
      },
      "boundary_MAE_seconds": {
        "mean": 0.03404335107272968,
        "CI95": [
          -0.10538475845730126,
          0.19090829863083236
        ],
        "queries": 128,
        "parents": 128,
        "improved": 22,
        "harmed": 13,
        "unchanged": 93,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 23,
      "severe_v": 21,
      "severe_union_once": 35,
      "rescue_union_once": 51
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 8,
      "severe_v": 6,
      "severe_union_once": 11,
      "rescue_union_once": 6
    },
    "Source-CleanSource": {
      "severe_t": 4,
      "severe_v": 3,
      "severe_union_once": 5,
      "rescue_union_once": 8
    }
  }
}
