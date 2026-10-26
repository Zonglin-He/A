# vidstg / legacy_Dim

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.13433186665259875,
      "t": 0.31104485241819246,
      "s": 0.36602265321430705,
      "fixed_actual_sIoU": 0.3700381068708722,
      "observed_actual_sIoU": 0.3937888865462764,
      "unobserved_actual_sIoU": 0.36879037230129164,
      "boundary_MAE": 216.44921875,
      "start_error_seconds": 6.923759611016582,
      "end_error_seconds": 8.36417479139704,
      "interval_length_seconds": 8.230420784181751,
      "length_error_seconds": 7.947851043738841,
      "vIoU_at_03": 0.1796875,
      "vIoU_at_05": 0.046875,
      "vIoU_ge_03": 0.1796875,
      "vIoU_ge_05": 0.046875,
      "tIoU_ge_05": 0.2734375,
      "tIoU_eq_0": 0.25,
      "boundary_MAE_seconds": 7.643967201206811
    },
    "FullMAP": {
      "v": 0.1515347738797456,
      "t": 0.34814750009505757,
      "s": 0.37109509830633947,
      "fixed_actual_sIoU": 0.374288781014965,
      "observed_actual_sIoU": 0.45841320374251787,
      "unobserved_actual_sIoU": 0.3672183033675137,
      "boundary_MAE": 203.90234375,
      "start_error_seconds": 6.843626838204708,
      "end_error_seconds": 7.5334443774213735,
      "interval_length_seconds": 11.442235516199483,
      "length_error_seconds": 8.035961974200026,
      "vIoU_at_03": 0.1875,
      "vIoU_at_05": 0.1171875,
      "vIoU_ge_03": 0.1875,
      "vIoU_ge_05": 0.1171875,
      "tIoU_ge_05": 0.3125,
      "tIoU_eq_0": 0.1796875,
      "boundary_MAE_seconds": 7.188535607813041
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
        "mean": 1.7202907227146844,
        "CI95": [
          -0.633298301389709,
          4.110176093478674
        ],
        "queries": 128,
        "parents": 128,
        "improved": 63,
        "harmed": 39,
        "unchanged": 26,
        "declined_over5pp": 25
      },
      "t": {
        "mean": 3.7102647676865077,
        "CI95": [
          1.435364926215839,
          6.157820552406757
        ],
        "queries": 128,
        "parents": 128,
        "improved": 61,
        "harmed": 38,
        "unchanged": 29,
        "declined_over5pp": 17
      },
      "observed_actual_sIoU": {
        "mean": 6.462431719624147,
        "CI95": [
          0.9016080029847611,
          11.945684114425053
        ],
        "queries": 105,
        "parents": 105,
        "improved": 65,
        "harmed": 39,
        "unchanged": 1,
        "declined_over5pp": 27
      },
      "unobserved_actual_sIoU": {
        "mean": -0.1572068933777958,
        "CI95": [
          -4.370117088608505,
          3.869631365091641
        ],
        "queries": 128,
        "parents": 128,
        "improved": 72,
        "harmed": 53,
        "unchanged": 3,
        "declined_over5pp": 37
      },
      "boundary_MAE_seconds": {
        "mean": -0.4554315933937708,
        "CI95": [
          -1.053218981844624,
          0.07110228604716153
        ],
        "queries": 128,
        "parents": 128,
        "improved": 64,
        "harmed": 51,
        "unchanged": 13,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.1857816527073094,
        "CI95": [
          -1.099440343244741,
          0.6786269764769022
        ],
        "queries": 128,
        "parents": 128,
        "improved": 49,
        "harmed": 51,
        "unchanged": 28,
        "declined_over5pp": 11
      },
      "t": {
        "mean": -0.3876847223347912,
        "CI95": [
          -2.3387683724422073,
          1.431837997333033
        ],
        "queries": 128,
        "parents": 128,
        "improved": 41,
        "harmed": 41,
        "unchanged": 46,
        "declined_over5pp": 16
      },
      "observed_actual_sIoU": {
        "mean": -1.647123126907444,
        "CI95": [
          -5.078016399662853,
          1.426315326996715
        ],
        "queries": 99,
        "parents": 99,
        "improved": 47,
        "harmed": 46,
        "unchanged": 6,
        "declined_over5pp": 19
      },
      "unobserved_actual_sIoU": {
        "mean": -1.4857443199334166,
        "CI95": [
          -3.6238167817008806,
          0.4834399143080688
        ],
        "queries": 128,
        "parents": 128,
        "improved": 56,
        "harmed": 57,
        "unchanged": 15,
        "declined_over5pp": 22
      },
      "boundary_MAE_seconds": {
        "mean": -0.08281393865091549,
        "CI95": [
          -0.48545199112275705,
          0.2886798885201033
        ],
        "queries": 128,
        "parents": 128,
        "improved": 48,
        "harmed": 49,
        "unchanged": 31,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": 0.11028393991663449,
        "CI95": [
          -0.7836114810085522,
          1.0283767867027116
        ],
        "queries": 128,
        "parents": 128,
        "improved": 48,
        "harmed": 47,
        "unchanged": 33,
        "declined_over5pp": 9
      },
      "t": {
        "mean": -1.0170495132399144,
        "CI95": [
          -3.2946660327656754,
          1.1600520635731713
        ],
        "queries": 128,
        "parents": 128,
        "improved": 36,
        "harmed": 42,
        "unchanged": 50,
        "declined_over5pp": 21
      },
      "observed_actual_sIoU": {
        "mean": 0.06226589288196999,
        "CI95": [
          -1.3196593944373458,
          1.4157404597340921
        ],
        "queries": 99,
        "parents": 99,
        "improved": 44,
        "harmed": 50,
        "unchanged": 5,
        "declined_over5pp": 11
      },
      "unobserved_actual_sIoU": {
        "mean": 0.46610104778299,
        "CI95": [
          -0.862874804182867,
          2.0204967794150384
        ],
        "queries": 128,
        "parents": 128,
        "improved": 57,
        "harmed": 66,
        "unchanged": 5,
        "declined_over5pp": 10
      },
      "boundary_MAE_seconds": {
        "mean": -0.0459493655150348,
        "CI95": [
          -0.3892797527768176,
          0.2650078008754179
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 54,
        "unchanged": 28,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 17,
      "severe_v": 25,
      "severe_union_once": 32,
      "rescue_union_once": 53
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 16,
      "severe_v": 11,
      "severe_union_once": 20,
      "rescue_union_once": 23
    },
    "Source-CleanSource": {
      "severe_t": 21,
      "severe_v": 9,
      "severe_union_once": 22,
      "rescue_union_once": 17
    }
  }
}
