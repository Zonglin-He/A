# vidstg / legacy_Blur

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.12297237773292662,
      "t": 0.318894878001491,
      "s": 0.34667733614454155,
      "fixed_actual_sIoU": 0.3521748784764075,
      "observed_actual_sIoU": 0.3609247825258534,
      "unobserved_actual_sIoU": 0.3501291661494822,
      "boundary_MAE": 217.66796875,
      "start_error_seconds": 6.978891621063946,
      "end_error_seconds": 8.428638152072713,
      "interval_length_seconds": 7.927770341553202,
      "length_error_seconds": 7.709417863417309,
      "vIoU_at_03": 0.1484375,
      "vIoU_at_05": 0.0234375,
      "vIoU_ge_03": 0.1484375,
      "vIoU_ge_05": 0.0234375,
      "tIoU_ge_05": 0.3125,
      "tIoU_eq_0": 0.2265625,
      "boundary_MAE_seconds": 7.703764886568331
    },
    "FullMAP": {
      "v": 0.14567138001827384,
      "t": 0.3502881528450751,
      "s": 0.3684573886030368,
      "fixed_actual_sIoU": 0.37196102215486104,
      "observed_actual_sIoU": 0.4262434180836694,
      "unobserved_actual_sIoU": 0.36801861985244977,
      "boundary_MAE": 205.9375,
      "start_error_seconds": 7.112232953802935,
      "end_error_seconds": 7.454490248392666,
      "interval_length_seconds": 11.45625370200113,
      "length_error_seconds": 8.526330789883403,
      "vIoU_at_03": 0.203125,
      "vIoU_at_05": 0.1171875,
      "vIoU_ge_03": 0.203125,
      "vIoU_ge_05": 0.1171875,
      "tIoU_ge_05": 0.3359375,
      "tIoU_eq_0": 0.171875,
      "boundary_MAE_seconds": 7.283361601097801
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
        "mean": 2.269900228534723,
        "CI95": [
          -0.10731780476524329,
          4.775574698305678
        ],
        "queries": 128,
        "parents": 128,
        "improved": 58,
        "harmed": 41,
        "unchanged": 29,
        "declined_over5pp": 23
      },
      "t": {
        "mean": 3.139327484358403,
        "CI95": [
          1.0396716775731067,
          5.450283664620776
        ],
        "queries": 128,
        "parents": 128,
        "improved": 66,
        "harmed": 35,
        "unchanged": 27,
        "declined_over5pp": 21
      },
      "observed_actual_sIoU": {
        "mean": 6.531863555781591,
        "CI95": [
          1.174438166997631,
          12.172755889755514
        ],
        "queries": 103,
        "parents": 103,
        "improved": 60,
        "harmed": 37,
        "unchanged": 6,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 1.7889453702967555,
        "CI95": [
          -2.513297579525195,
          6.126236933561079
        ],
        "queries": 128,
        "parents": 128,
        "improved": 73,
        "harmed": 50,
        "unchanged": 5,
        "declined_over5pp": 33
      },
      "boundary_MAE_seconds": {
        "mean": -0.4204032854705284,
        "CI95": [
          -1.0273034524765023,
          0.11298905105313195
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
        "mean": -0.7721210388544837,
        "CI95": [
          -2.2224276849851323,
          0.8082962106821836
        ],
        "queries": 128,
        "parents": 128,
        "improved": 40,
        "harmed": 60,
        "unchanged": 28,
        "declined_over5pp": 18
      },
      "t": {
        "mean": -0.1736194473330419,
        "CI95": [
          -1.909568331998014,
          1.3748037707936267
        ],
        "queries": 128,
        "parents": 128,
        "improved": 38,
        "harmed": 41,
        "unchanged": 49,
        "declined_over5pp": 16
      },
      "observed_actual_sIoU": {
        "mean": -2.8370700031910663,
        "CI95": [
          -6.7029152446720435,
          0.6797518891895314
        ],
        "queries": 100,
        "parents": 100,
        "improved": 38,
        "harmed": 53,
        "unchanged": 9,
        "declined_over5pp": 17
      },
      "unobserved_actual_sIoU": {
        "mean": -1.4057126714398143,
        "CI95": [
          -4.092284664928278,
          1.1472060286798835
        ],
        "queries": 128,
        "parents": 128,
        "improved": 51,
        "harmed": 64,
        "unchanged": 13,
        "declined_over5pp": 23
      },
      "boundary_MAE_seconds": {
        "mean": 0.01201205463384497,
        "CI95": [
          -0.35844298200792407,
          0.4024772246916375
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 47,
        "unchanged": 35,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -1.0256649520505787,
        "CI95": [
          -2.090073866290462,
          0.00999994919623225
        ],
        "queries": 128,
        "parents": 128,
        "improved": 36,
        "harmed": 60,
        "unchanged": 32,
        "declined_over5pp": 15
      },
      "t": {
        "mean": -0.23204695491006122,
        "CI95": [
          -2.070638992184789,
          1.6386615797167334
        ],
        "queries": 128,
        "parents": 128,
        "improved": 32,
        "harmed": 38,
        "unchanged": 58,
        "declined_over5pp": 22
      },
      "observed_actual_sIoU": {
        "mean": -1.4500431733321932,
        "CI95": [
          -3.5175426684075233,
          0.4069587452746944
        ],
        "queries": 100,
        "parents": 100,
        "improved": 38,
        "harmed": 53,
        "unchanged": 9,
        "declined_over5pp": 11
      },
      "unobserved_actual_sIoU": {
        "mean": -1.4000195673979579,
        "CI95": [
          -3.0195910837089612,
          0.024709894903489463
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 76,
        "unchanged": 6,
        "declined_over5pp": 10
      },
      "boundary_MAE_seconds": {
        "mean": 0.013848319846483256,
        "CI95": [
          -0.23218118411556307,
          0.23751798818666633
        ],
        "queries": 128,
        "parents": 128,
        "improved": 45,
        "harmed": 44,
        "unchanged": 39,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 21,
      "severe_v": 23,
      "severe_union_once": 35,
      "rescue_union_once": 57
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 16,
      "severe_v": 18,
      "severe_union_once": 25,
      "rescue_union_once": 26
    },
    "Source-CleanSource": {
      "severe_t": 22,
      "severe_v": 15,
      "severe_union_once": 28,
      "rescue_union_once": 16
    }
  }
}
