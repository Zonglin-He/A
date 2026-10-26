# vidstg / motion_blur_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.1339840433971608,
      "t": 0.3179706595370419,
      "s": 0.35989341255740875,
      "fixed_actual_sIoU": 0.3637779960450292,
      "observed_actual_sIoU": 0.35808049545595955,
      "unobserved_actual_sIoU": 0.36459255618449093,
      "boundary_MAE": 215.5703125,
      "start_error_seconds": 6.83086755942252,
      "end_error_seconds": 8.385572413627067,
      "interval_length_seconds": 7.558903447525943,
      "length_error_seconds": 7.5687651091041825,
      "vIoU_at_03": 0.171875,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.171875,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.2890625,
      "tIoU_eq_0": 0.2109375,
      "boundary_MAE_seconds": 7.608219986524793
    },
    "FullMAP": {
      "v": 0.15172579260793653,
      "t": 0.3493413349433262,
      "s": 0.38498257814188797,
      "fixed_actual_sIoU": 0.39042549168734975,
      "observed_actual_sIoU": 0.43803289209144947,
      "unobserved_actual_sIoU": 0.38585275961316157,
      "boundary_MAE": 205.09765625,
      "start_error_seconds": 7.0353585965024,
      "end_error_seconds": 7.481213424670246,
      "interval_length_seconds": 11.145095264722771,
      "length_error_seconds": 8.06796122394012,
      "vIoU_at_03": 0.1875,
      "vIoU_at_05": 0.1015625,
      "vIoU_ge_03": 0.1875,
      "vIoU_ge_05": 0.1015625,
      "tIoU_ge_05": 0.3359375,
      "tIoU_eq_0": 0.1640625,
      "boundary_MAE_seconds": 7.258286010586323
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
        "mean": 1.7741749210775752,
        "CI95": [
          -0.22529706802230554,
          3.763916540006834
        ],
        "queries": 128,
        "parents": 128,
        "improved": 67,
        "harmed": 36,
        "unchanged": 25,
        "declined_over5pp": 20
      },
      "t": {
        "mean": 3.1370675406284274,
        "CI95": [
          1.1851836769361122,
          5.275516392901923
        ],
        "queries": 128,
        "parents": 128,
        "improved": 64,
        "harmed": 35,
        "unchanged": 29,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": 7.995239663548995,
        "CI95": [
          2.7792959949289777,
          13.193527031126807
        ],
        "queries": 103,
        "parents": 103,
        "improved": 67,
        "harmed": 30,
        "unchanged": 6,
        "declined_over5pp": 22
      },
      "unobserved_actual_sIoU": {
        "mean": 2.126020342867062,
        "CI95": [
          -1.782998667070911,
          5.963469367293643
        ],
        "queries": 128,
        "parents": 128,
        "improved": 71,
        "harmed": 53,
        "unchanged": 4,
        "declined_over5pp": 34
      },
      "boundary_MAE_seconds": {
        "mean": -0.34993397593847025,
        "CI95": [
          -0.9790234841265286,
          0.1895478082056092
        ],
        "queries": 128,
        "parents": 128,
        "improved": 63,
        "harmed": 54,
        "unchanged": 11,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.166679779888215,
        "CI95": [
          -0.9235798586913352,
          0.5699234247878877
        ],
        "queries": 128,
        "parents": 128,
        "improved": 39,
        "harmed": 32,
        "unchanged": 57,
        "declined_over5pp": 9
      },
      "t": {
        "mean": -0.2683012375079321,
        "CI95": [
          -1.4318178185962558,
          0.8051030921436045
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 27,
        "unchanged": 73,
        "declined_over5pp": 9
      },
      "observed_actual_sIoU": {
        "mean": -0.4110445642538788,
        "CI95": [
          -1.8759001183845976,
          1.0626955727699827
        ],
        "queries": 101,
        "parents": 101,
        "improved": 24,
        "harmed": 40,
        "unchanged": 37,
        "declined_over5pp": 10
      },
      "unobserved_actual_sIoU": {
        "mean": 0.3777013046313695,
        "CI95": [
          -0.6311640073903289,
          1.5717326020237752
        ],
        "queries": 128,
        "parents": 128,
        "improved": 40,
        "harmed": 43,
        "unchanged": 45,
        "declined_over5pp": 9
      },
      "boundary_MAE_seconds": {
        "mean": -0.013063535877632954,
        "CI95": [
          -0.2217938105400578,
          0.21063236666275037
        ],
        "queries": 128,
        "parents": 128,
        "improved": 36,
        "harmed": 31,
        "unchanged": 61,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": 0.07550161437283773,
        "CI95": [
          -0.5826435198780799,
          0.8266356298632814
        ],
        "queries": 128,
        "parents": 128,
        "improved": 27,
        "harmed": 41,
        "unchanged": 60,
        "declined_over5pp": 4
      },
      "t": {
        "mean": -0.32446880135497574,
        "CI95": [
          -1.7186477600220826,
          1.1168188077194674
        ],
        "queries": 128,
        "parents": 128,
        "improved": 20,
        "harmed": 26,
        "unchanged": 82,
        "declined_over5pp": 13
      },
      "observed_actual_sIoU": {
        "mean": -0.9576461594286154,
        "CI95": [
          -2.2026144331306483,
          -0.028338711863025053
        ],
        "queries": 101,
        "parents": 101,
        "improved": 26,
        "harmed": 38,
        "unchanged": 37,
        "declined_over5pp": 7
      },
      "unobserved_actual_sIoU": {
        "mean": 0.046319436102919415,
        "CI95": [
          -0.4568414792096449,
          0.49328412510807856
        ],
        "queries": 128,
        "parents": 128,
        "improved": 51,
        "harmed": 41,
        "unchanged": 36,
        "declined_over5pp": 2
      },
      "boundary_MAE_seconds": {
        "mean": -0.08169658019705288,
        "CI95": [
          -0.35152702230475513,
          0.18334224111393713
        ],
        "queries": 128,
        "parents": 128,
        "improved": 29,
        "harmed": 33,
        "unchanged": 66,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 20,
      "severe_v": 20,
      "severe_union_once": 33,
      "rescue_union_once": 56
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 9,
      "severe_v": 9,
      "severe_union_once": 13,
      "rescue_union_once": 12
    },
    "Source-CleanSource": {
      "severe_t": 13,
      "severe_v": 4,
      "severe_union_once": 13,
      "rescue_union_once": 12
    }
  }
}
