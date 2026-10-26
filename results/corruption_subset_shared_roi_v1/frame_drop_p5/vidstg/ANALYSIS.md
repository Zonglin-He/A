# vidstg / frame_drop_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.12890004897106205,
      "t": 0.3142957685708684,
      "s": 0.3547031984490041,
      "fixed_actual_sIoU": 0.3587411195191872,
      "observed_actual_sIoU": 0.35754198130202064,
      "unobserved_actual_sIoU": 0.3596939610729075,
      "boundary_MAE": 218.0078125,
      "start_error_seconds": 6.771226000474441,
      "end_error_seconds": 8.621079290278072,
      "interval_length_seconds": 7.395379781412439,
      "length_error_seconds": 7.722199183063363,
      "vIoU_at_03": 0.1484375,
      "vIoU_at_05": 0.03125,
      "vIoU_ge_03": 0.1484375,
      "vIoU_ge_05": 0.03125,
      "tIoU_ge_05": 0.2734375,
      "tIoU_eq_0": 0.21875,
      "boundary_MAE_seconds": 7.696152645376255
    },
    "FullMAP": {
      "v": 0.14376567897446596,
      "t": 0.3443635515263685,
      "s": 0.3718564134738455,
      "fixed_actual_sIoU": 0.377222220439714,
      "observed_actual_sIoU": 0.4308640656344158,
      "unobserved_actual_sIoU": 0.37191740632861214,
      "boundary_MAE": 211.01953125,
      "start_error_seconds": 7.232030425167002,
      "end_error_seconds": 7.715214437183091,
      "interval_length_seconds": 10.712035731079588,
      "length_error_seconds": 7.9811037265759,
      "vIoU_at_03": 0.1875,
      "vIoU_at_05": 0.0859375,
      "vIoU_ge_03": 0.1875,
      "vIoU_ge_05": 0.0859375,
      "tIoU_ge_05": 0.328125,
      "tIoU_eq_0": 0.1796875,
      "boundary_MAE_seconds": 7.4736224311750465
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
        "mean": 1.4865630003403933,
        "CI95": [
          -0.5546243630252138,
          3.5398206535148207
        ],
        "queries": 128,
        "parents": 128,
        "improved": 62,
        "harmed": 40,
        "unchanged": 26,
        "declined_over5pp": 21
      },
      "t": {
        "mean": 3.006778295550011,
        "CI95": [
          0.793402766071911,
          5.546759001088653
        ],
        "queries": 128,
        "parents": 128,
        "improved": 62,
        "harmed": 35,
        "unchanged": 31,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": 7.3322084332395185,
        "CI95": [
          1.8654071815606281,
          12.858632477355936
        ],
        "queries": 102,
        "parents": 102,
        "improved": 66,
        "harmed": 32,
        "unchanged": 4,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 1.2223445255704712,
        "CI95": [
          -2.835734201480356,
          5.249707718520114
        ],
        "queries": 128,
        "parents": 128,
        "improved": 71,
        "harmed": 54,
        "unchanged": 3,
        "declined_over5pp": 34
      },
      "boundary_MAE_seconds": {
        "mean": -0.22253021420120916,
        "CI95": [
          -0.8467581665423972,
          0.32733929383592386
        ],
        "queries": 128,
        "parents": 128,
        "improved": 60,
        "harmed": 56,
        "unchanged": 12,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.9626911432352727,
        "CI95": [
          -2.174380643232934,
          0.09451081572182327
        ],
        "queries": 128,
        "parents": 128,
        "improved": 33,
        "harmed": 37,
        "unchanged": 58,
        "declined_over5pp": 15
      },
      "t": {
        "mean": -0.7660795792036996,
        "CI95": [
          -2.2630485365257913,
          0.5788342042532365
        ],
        "queries": 128,
        "parents": 128,
        "improved": 26,
        "harmed": 33,
        "unchanged": 69,
        "declined_over5pp": 17
      },
      "observed_actual_sIoU": {
        "mean": -2.3180486048489524,
        "CI95": [
          -4.926843395282575,
          -0.19790913930242135
        ],
        "queries": 99,
        "parents": 99,
        "improved": 29,
        "harmed": 36,
        "unchanged": 34,
        "declined_over5pp": 15
      },
      "unobserved_actual_sIoU": {
        "mean": -1.0158340238235735,
        "CI95": [
          -3.2160406425148107,
          0.9409622658435305
        ],
        "queries": 128,
        "parents": 128,
        "improved": 40,
        "harmed": 44,
        "unchanged": 44,
        "declined_over5pp": 13
      },
      "boundary_MAE_seconds": {
        "mean": 0.20227288471109037,
        "CI95": [
          -0.10757945277308548,
          0.5633801142597257
        ],
        "queries": 128,
        "parents": 128,
        "improved": 33,
        "harmed": 39,
        "unchanged": 56,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.4328978282370385,
        "CI95": [
          -1.0856543327378758,
          0.20651184484183496
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 38,
        "unchanged": 60,
        "declined_over5pp": 7
      },
      "t": {
        "mean": -0.6919578979723263,
        "CI95": [
          -1.916014193548112,
          0.4572084490082235
        ],
        "queries": 128,
        "parents": 128,
        "improved": 27,
        "harmed": 27,
        "unchanged": 74,
        "declined_over5pp": 14
      },
      "observed_actual_sIoU": {
        "mean": -1.7296705621173591,
        "CI95": [
          -3.200362000112027,
          -0.6018043097362902
        ],
        "queries": 99,
        "parents": 99,
        "improved": 25,
        "harmed": 37,
        "unchanged": 37,
        "declined_over5pp": 9
      },
      "unobserved_actual_sIoU": {
        "mean": -0.4435400750554325,
        "CI95": [
          -1.1588877714483252,
          0.1905922181587279
        ],
        "queries": 128,
        "parents": 128,
        "improved": 39,
        "harmed": 53,
        "unchanged": 36,
        "declined_over5pp": 6
      },
      "boundary_MAE_seconds": {
        "mean": 0.006236078654409418,
        "CI95": [
          -0.22883881273321702,
          0.25417018989779766
        ],
        "queries": 128,
        "parents": 128,
        "improved": 36,
        "harmed": 34,
        "unchanged": 58,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 20,
      "severe_v": 21,
      "severe_union_once": 32,
      "rescue_union_once": 55
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 17,
      "severe_v": 15,
      "severe_union_once": 21,
      "rescue_union_once": 14
    },
    "Source-CleanSource": {
      "severe_t": 14,
      "severe_v": 7,
      "severe_union_once": 15,
      "rescue_union_once": 10
    }
  }
}
