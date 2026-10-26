# vidstg / motion_blur_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.13561942005336847,
      "t": 0.32448712551565084,
      "s": 0.3607349649903111,
      "fixed_actual_sIoU": 0.36462081595962215,
      "observed_actual_sIoU": 0.37090713485639554,
      "unobserved_actual_sIoU": 0.3654287478453987,
      "boundary_MAE": 214.21484375,
      "start_error_seconds": 6.856605325813747,
      "end_error_seconds": 8.25797190687031,
      "interval_length_seconds": 7.774795770208167,
      "length_error_seconds": 7.544295951556176,
      "vIoU_at_03": 0.171875,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.171875,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.3046875,
      "tIoU_eq_0": 0.2109375,
      "boundary_MAE_seconds": 7.557288616342028
    },
    "FullMAP": {
      "v": 0.15169438091785103,
      "t": 0.35589458996705603,
      "s": 0.37861840383130296,
      "fixed_actual_sIoU": 0.3820230779347756,
      "observed_actual_sIoU": 0.44962479675822925,
      "unobserved_actual_sIoU": 0.3762138374852243,
      "boundary_MAE": 202.765625,
      "start_error_seconds": 7.029839843374272,
      "end_error_seconds": 7.344291155917118,
      "interval_length_seconds": 11.451135626177976,
      "length_error_seconds": 8.082966946850531,
      "vIoU_at_03": 0.203125,
      "vIoU_at_05": 0.109375,
      "vIoU_ge_03": 0.203125,
      "vIoU_ge_05": 0.109375,
      "tIoU_ge_05": 0.34375,
      "tIoU_eq_0": 0.1640625,
      "boundary_MAE_seconds": 7.187065499645694
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
        "mean": 1.6074960864482546,
        "CI95": [
          -0.4661204730508154,
          3.7193172733181012
        ],
        "queries": 128,
        "parents": 128,
        "improved": 65,
        "harmed": 38,
        "unchanged": 25,
        "declined_over5pp": 21
      },
      "t": {
        "mean": 3.140746445140514,
        "CI95": [
          0.7515072553707922,
          5.776031976374257
        ],
        "queries": 128,
        "parents": 128,
        "improved": 55,
        "harmed": 41,
        "unchanged": 32,
        "declined_over5pp": 22
      },
      "observed_actual_sIoU": {
        "mean": 7.871766190183371,
        "CI95": [
          2.5669196229436864,
          13.135628038689596
        ],
        "queries": 101,
        "parents": 101,
        "improved": 69,
        "harmed": 29,
        "unchanged": 3,
        "declined_over5pp": 23
      },
      "unobserved_actual_sIoU": {
        "mean": 1.0785089639825665,
        "CI95": [
          -2.9112251923881676,
          4.95266460350429
        ],
        "queries": 128,
        "parents": 128,
        "improved": 73,
        "harmed": 52,
        "unchanged": 3,
        "declined_over5pp": 34
      },
      "boundary_MAE_seconds": {
        "mean": -0.3702231166963328,
        "CI95": [
          -1.0517823971123543,
          0.21414482562800363
        ],
        "queries": 128,
        "parents": 128,
        "improved": 57,
        "harmed": 57,
        "unchanged": 14,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.16982094889676666,
        "CI95": [
          -0.9170366298068751,
          0.46991854583834974
        ],
        "queries": 128,
        "parents": 128,
        "improved": 34,
        "harmed": 36,
        "unchanged": 58,
        "declined_over5pp": 6
      },
      "t": {
        "mean": 0.3870242648650532,
        "CI95": [
          -0.7056456547228319,
          1.5191639390614116
        ],
        "queries": 128,
        "parents": 128,
        "improved": 24,
        "harmed": 22,
        "unchanged": 82,
        "declined_over5pp": 7
      },
      "observed_actual_sIoU": {
        "mean": 0.06072471960574876,
        "CI95": [
          -0.6441060534381331,
          0.716239547823265
        ],
        "queries": 99,
        "parents": 99,
        "improved": 26,
        "harmed": 34,
        "unchanged": 39,
        "declined_over5pp": 5
      },
      "unobserved_actual_sIoU": {
        "mean": -0.5861909081623569,
        "CI95": [
          -1.2997421990886235,
          0.005560110231582384
        ],
        "queries": 128,
        "parents": 128,
        "improved": 37,
        "harmed": 46,
        "unchanged": 45,
        "declined_over5pp": 10
      },
      "boundary_MAE_seconds": {
        "mean": -0.08428404681826115,
        "CI95": [
          -0.24772239345345579,
          0.07987999649390287
        ],
        "queries": 128,
        "parents": 128,
        "improved": 29,
        "harmed": 29,
        "unchanged": 70,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": 0.23903927999360647,
        "CI95": [
          -0.0966549614387275,
          0.6756698882149713
        ],
        "queries": 128,
        "parents": 128,
        "improved": 29,
        "harmed": 37,
        "unchanged": 62,
        "declined_over5pp": 1
      },
      "t": {
        "mean": 0.32717779650592316,
        "CI95": [
          -0.7093993493361813,
          1.488333247922596
        ],
        "queries": 128,
        "parents": 128,
        "improved": 17,
        "harmed": 20,
        "unchanged": 91,
        "declined_over5pp": 6
      },
      "observed_actual_sIoU": {
        "mean": -0.5673453623055387,
        "CI95": [
          -1.5327005260718254,
          0.07887859038599948
        ],
        "queries": 99,
        "parents": 99,
        "improved": 28,
        "harmed": 33,
        "unchanged": 38,
        "declined_over5pp": 4
      },
      "unobserved_actual_sIoU": {
        "mean": 0.12993860219368897,
        "CI95": [
          -0.17745596834536437,
          0.46570642769867227
        ],
        "queries": 128,
        "parents": 128,
        "improved": 50,
        "harmed": 41,
        "unchanged": 37,
        "declined_over5pp": 1
      },
      "boundary_MAE_seconds": {
        "mean": -0.1326279503798185,
        "CI95": [
          -0.29597356426195615,
          0.011066712011277237
        ],
        "queries": 128,
        "parents": 128,
        "improved": 25,
        "harmed": 25,
        "unchanged": 78,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 22,
      "severe_v": 21,
      "severe_union_once": 34,
      "rescue_union_once": 55
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 7,
      "severe_v": 6,
      "severe_union_once": 9,
      "rescue_union_once": 12
    },
    "Source-CleanSource": {
      "severe_t": 6,
      "severe_v": 1,
      "severe_union_once": 6,
      "rescue_union_once": 9
    }
  }
}
