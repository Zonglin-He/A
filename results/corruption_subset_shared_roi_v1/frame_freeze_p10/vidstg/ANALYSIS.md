# vidstg / frame_freeze_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.1256843626944605,
      "t": 0.3152868305299672,
      "s": 0.34416768670937065,
      "fixed_actual_sIoU": 0.3487986563685478,
      "observed_actual_sIoU": 0.35214157251621975,
      "unobserved_actual_sIoU": 0.34999530292151726,
      "boundary_MAE": 218.3125,
      "start_error_seconds": 7.099418007757571,
      "end_error_seconds": 8.319377629833305,
      "interval_length_seconds": 7.682091387877783,
      "length_error_seconds": 7.6846663064598655,
      "vIoU_at_03": 0.1484375,
      "vIoU_at_05": 0.03125,
      "vIoU_ge_03": 0.1484375,
      "vIoU_ge_05": 0.03125,
      "tIoU_ge_05": 0.28125,
      "tIoU_eq_0": 0.2265625,
      "boundary_MAE_seconds": 7.709397818795436
    },
    "FullMAP": {
      "v": 0.14478088676042136,
      "t": 0.3426190005200468,
      "s": 0.36611059177088834,
      "fixed_actual_sIoU": 0.37280895408126485,
      "observed_actual_sIoU": 0.4316562801001676,
      "unobserved_actual_sIoU": 0.36795833296188063,
      "boundary_MAE": 210.38671875,
      "start_error_seconds": 7.28315736147673,
      "end_error_seconds": 7.576593218514544,
      "interval_length_seconds": 11.116031829840246,
      "length_error_seconds": 8.462596095016487,
      "vIoU_at_03": 0.1796875,
      "vIoU_at_05": 0.09375,
      "vIoU_ge_03": 0.1796875,
      "vIoU_ge_05": 0.09375,
      "tIoU_ge_05": 0.3125,
      "tIoU_eq_0": 0.1796875,
      "boundary_MAE_seconds": 7.429875289995637
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
        "mean": 1.9096524065960856,
        "CI95": [
          -0.16158644854781623,
          4.012663272107858
        ],
        "queries": 128,
        "parents": 128,
        "improved": 64,
        "harmed": 36,
        "unchanged": 28,
        "declined_over5pp": 19
      },
      "t": {
        "mean": 2.7332169990079613,
        "CI95": [
          0.4899223952262968,
          5.319803108245163
        ],
        "queries": 128,
        "parents": 128,
        "improved": 50,
        "harmed": 46,
        "unchanged": 32,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": 7.951470758394782,
        "CI95": [
          2.7605592656957954,
          13.186512189426525
        ],
        "queries": 104,
        "parents": 104,
        "improved": 69,
        "harmed": 30,
        "unchanged": 5,
        "declined_over5pp": 26
      },
      "unobserved_actual_sIoU": {
        "mean": 1.7963030040363355,
        "CI95": [
          -2.425803879835337,
          5.864733251433925
        ],
        "queries": 128,
        "parents": 128,
        "improved": 76,
        "harmed": 49,
        "unchanged": 3,
        "declined_over5pp": 33
      },
      "boundary_MAE_seconds": {
        "mean": -0.27952252879979905,
        "CI95": [
          -0.9329735694064508,
          0.28515834886651537
        ],
        "queries": 128,
        "parents": 128,
        "improved": 55,
        "harmed": 61,
        "unchanged": 12,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.8611703646397332,
        "CI95": [
          -1.9790100236166792,
          0.03314077118375271
        ],
        "queries": 128,
        "parents": 128,
        "improved": 34,
        "harmed": 38,
        "unchanged": 56,
        "declined_over5pp": 8
      },
      "t": {
        "mean": -0.9405346798358722,
        "CI95": [
          -2.0235662404593753,
          0.06170500146846894
        ],
        "queries": 128,
        "parents": 128,
        "improved": 25,
        "harmed": 26,
        "unchanged": 77,
        "declined_over5pp": 13
      },
      "observed_actual_sIoU": {
        "mean": -2.3400861729930393,
        "CI95": [
          -5.3031408496829355,
          0.19731262823593496
        ],
        "queries": 101,
        "parents": 101,
        "improved": 34,
        "harmed": 33,
        "unchanged": 34,
        "declined_over5pp": 11
      },
      "unobserved_actual_sIoU": {
        "mean": -1.4117413604967275,
        "CI95": [
          -3.79501775550648,
          0.7904648992942309
        ],
        "queries": 128,
        "parents": 128,
        "improved": 39,
        "harmed": 49,
        "unchanged": 40,
        "declined_over5pp": 15
      },
      "boundary_MAE_seconds": {
        "mean": 0.1585257435316819,
        "CI95": [
          -0.12815065904942707,
          0.4734142763246645
        ],
        "queries": 128,
        "parents": 128,
        "improved": 31,
        "harmed": 34,
        "unchanged": 63,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.7544664558971905,
        "CI95": [
          -1.5624337437706584,
          -0.03054015157742967
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 38,
        "unchanged": 60,
        "declined_over5pp": 7
      },
      "t": {
        "mean": -0.5928517020624499,
        "CI95": [
          -1.975245519063238,
          0.7475432847741336
        ],
        "queries": 128,
        "parents": 128,
        "improved": 23,
        "harmed": 21,
        "unchanged": 84,
        "declined_over5pp": 12
      },
      "observed_actual_sIoU": {
        "mean": -1.9694405837970135,
        "CI95": [
          -3.876740835739397,
          -0.5388775720750224
        ],
        "queries": 101,
        "parents": 101,
        "improved": 29,
        "harmed": 39,
        "unchanged": 33,
        "declined_over5pp": 7
      },
      "unobserved_actual_sIoU": {
        "mean": -1.4134058901944515,
        "CI95": [
          -3.226812744559792,
          0.043233109125532534
        ],
        "queries": 128,
        "parents": 128,
        "improved": 44,
        "harmed": 51,
        "unchanged": 33,
        "declined_over5pp": 8
      },
      "boundary_MAE_seconds": {
        "mean": 0.01948125207359075,
        "CI95": [
          -0.3314288984006054,
          0.3333534248237297
        ],
        "queries": 128,
        "parents": 128,
        "improved": 33,
        "harmed": 25,
        "unchanged": 70,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 20,
      "severe_v": 19,
      "severe_union_once": 33,
      "rescue_union_once": 49
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 13,
      "severe_v": 8,
      "severe_union_once": 16,
      "rescue_union_once": 7
    },
    "Source-CleanSource": {
      "severe_t": 12,
      "severe_v": 7,
      "severe_union_once": 13,
      "rescue_union_once": 7
    }
  }
}
