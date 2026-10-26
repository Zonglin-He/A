# vidstg / frame_drop_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.12955781270508832,
      "t": 0.3173773220225615,
      "s": 0.35735869740617054,
      "fixed_actual_sIoU": 0.36149414001294655,
      "observed_actual_sIoU": 0.3616596613333459,
      "unobserved_actual_sIoU": 0.3627529817367105,
      "boundary_MAE": 217.20703125,
      "start_error_seconds": 6.8137099892599355,
      "end_error_seconds": 8.528735301405636,
      "interval_length_seconds": 7.6765158294223035,
      "length_error_seconds": 7.847356742896048,
      "vIoU_at_03": 0.1484375,
      "vIoU_at_05": 0.03125,
      "vIoU_ge_03": 0.1484375,
      "vIoU_ge_05": 0.03125,
      "tIoU_ge_05": 0.28125,
      "tIoU_eq_0": 0.21875,
      "boundary_MAE_seconds": 7.671222645332786
    },
    "FullMAP": {
      "v": 0.15146625278989567,
      "t": 0.35239473703163815,
      "s": 0.3783769282415208,
      "fixed_actual_sIoU": 0.3830219712882158,
      "observed_actual_sIoU": 0.44711228649980794,
      "unobserved_actual_sIoU": 0.37767170480833584,
      "boundary_MAE": 205.3359375,
      "start_error_seconds": 7.058003794434707,
      "end_error_seconds": 7.525349982494027,
      "interval_length_seconds": 11.111454684212408,
      "length_error_seconds": 8.283212380737876,
      "vIoU_at_03": 0.1875,
      "vIoU_at_05": 0.09375,
      "vIoU_ge_03": 0.1875,
      "vIoU_ge_05": 0.09375,
      "tIoU_ge_05": 0.3359375,
      "tIoU_eq_0": 0.1640625,
      "boundary_MAE_seconds": 7.291676888464368
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
        "mean": 2.190844008480739,
        "CI95": [
          0.11316828414979181,
          4.308921609673127
        ],
        "queries": 128,
        "parents": 128,
        "improved": 67,
        "harmed": 36,
        "unchanged": 25,
        "declined_over5pp": 20
      },
      "t": {
        "mean": 3.5017415009076567,
        "CI95": [
          1.2601736313417948,
          6.006940652182667
        ],
        "queries": 128,
        "parents": 128,
        "improved": 62,
        "harmed": 38,
        "unchanged": 28,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 8.545262516646202,
        "CI95": [
          3.251563725378109,
          13.829807412829428
        ],
        "queries": 100,
        "parents": 100,
        "improved": 68,
        "harmed": 28,
        "unchanged": 4,
        "declined_over5pp": 23
      },
      "unobserved_actual_sIoU": {
        "mean": 1.4918723071625337,
        "CI95": [
          -2.4199926304234296,
          5.293045354772967
        ],
        "queries": 128,
        "parents": 128,
        "improved": 73,
        "harmed": 52,
        "unchanged": 3,
        "declined_over5pp": 34
      },
      "boundary_MAE_seconds": {
        "mean": -0.3795457568684183,
        "CI95": [
          -1.0035353358348935,
          0.1525769010391971
        ],
        "queries": 128,
        "parents": 128,
        "improved": 62,
        "harmed": 54,
        "unchanged": 12,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.19263376169229873,
        "CI95": [
          -0.9541976997665399,
          0.5543537757631242
        ],
        "queries": 128,
        "parents": 128,
        "improved": 38,
        "harmed": 31,
        "unchanged": 59,
        "declined_over5pp": 7
      },
      "t": {
        "mean": 0.037038971323260894,
        "CI95": [
          -1.1689306775109165,
          1.2578176212288101
        ],
        "queries": 128,
        "parents": 128,
        "improved": 27,
        "harmed": 27,
        "unchanged": 74,
        "declined_over5pp": 11
      },
      "observed_actual_sIoU": {
        "mean": -0.5449562452741812,
        "CI95": [
          -1.4206022797684479,
          0.3240622136476459
        ],
        "queries": 97,
        "parents": 97,
        "improved": 23,
        "harmed": 34,
        "unchanged": 40,
        "declined_over5pp": 8
      },
      "unobserved_actual_sIoU": {
        "mean": -0.4404041758512052,
        "CI95": [
          -1.6093875847781223,
          0.7855649660388695
        ],
        "queries": 128,
        "parents": 128,
        "improved": 37,
        "harmed": 46,
        "unchanged": 45,
        "declined_over5pp": 10
      },
      "boundary_MAE_seconds": {
        "mean": 0.02032734200041099,
        "CI95": [
          -0.2023913855230006,
          0.2972870087768537
        ],
        "queries": 128,
        "parents": 128,
        "improved": 33,
        "harmed": 34,
        "unchanged": 61,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.3671214548344099,
        "CI95": [
          -0.9240712164820278,
          0.1346798145233321
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 36,
        "unchanged": 62,
        "declined_over5pp": 6
      },
      "t": {
        "mean": -0.38380255280301234,
        "CI95": [
          -1.3216730210630367,
          0.4996382280270949
        ],
        "queries": 128,
        "parents": 128,
        "improved": 18,
        "harmed": 25,
        "unchanged": 85,
        "declined_over5pp": 11
      },
      "observed_actual_sIoU": {
        "mean": -1.4018015838417561,
        "CI95": [
          -2.6196159602392877,
          -0.49558978044570107
        ],
        "queries": 97,
        "parents": 97,
        "improved": 20,
        "harmed": 39,
        "unchanged": 38,
        "declined_over5pp": 7
      },
      "unobserved_actual_sIoU": {
        "mean": -0.1376380086751277,
        "CI95": [
          -0.6514952565491425,
          0.34648445453563725
        ],
        "queries": 128,
        "parents": 128,
        "improved": 48,
        "harmed": 43,
        "unchanged": 37,
        "declined_over5pp": 4
      },
      "boundary_MAE_seconds": {
        "mean": -0.0186939213890609,
        "CI95": [
          -0.21559823365463557,
          0.1889165976457172
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 30,
        "unchanged": 70,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 23,
      "severe_v": 20,
      "severe_union_once": 35,
      "rescue_union_once": 57
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 11,
      "severe_v": 7,
      "severe_union_once": 13,
      "rescue_union_once": 10
    },
    "Source-CleanSource": {
      "severe_t": 11,
      "severe_v": 6,
      "severe_union_once": 12,
      "rescue_union_once": 11
    }
  }
}
