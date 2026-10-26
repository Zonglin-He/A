# vidstg / occlusion_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.13046878350985724,
      "t": 0.3209252189948263,
      "s": 0.3531979805144052,
      "fixed_actual_sIoU": 0.3577047597522618,
      "observed_actual_sIoU": 0.36190259243530315,
      "unobserved_actual_sIoU": 0.35648552487656626,
      "boundary_MAE": 211.5859375,
      "start_error_seconds": 6.759763611247617,
      "end_error_seconds": 8.163369791044296,
      "interval_length_seconds": 7.491050194529677,
      "length_error_seconds": 7.593445251403649,
      "vIoU_at_03": 0.15625,
      "vIoU_at_05": 0.046875,
      "vIoU_ge_03": 0.15625,
      "vIoU_ge_05": 0.046875,
      "tIoU_ge_05": 0.2890625,
      "tIoU_eq_0": 0.234375,
      "boundary_MAE_seconds": 7.461566701145957
    },
    "FullMAP": {
      "v": 0.14667979400300923,
      "t": 0.34698473931640683,
      "s": 0.3797025095496882,
      "fixed_actual_sIoU": 0.38501952469251366,
      "observed_actual_sIoU": 0.4409648686950719,
      "unobserved_actual_sIoU": 0.3794330151556544,
      "boundary_MAE": 202.16015625,
      "start_error_seconds": 6.7778150359296925,
      "end_error_seconds": 7.524757310745401,
      "interval_length_seconds": 10.580593732935913,
      "length_error_seconds": 8.10628668590716,
      "vIoU_at_03": 0.1953125,
      "vIoU_at_05": 0.09375,
      "vIoU_ge_03": 0.1953125,
      "vIoU_ge_05": 0.09375,
      "tIoU_ge_05": 0.3203125,
      "tIoU_eq_0": 0.1796875,
      "boundary_MAE_seconds": 7.151286173337548
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
        "mean": 1.621101049315199,
        "CI95": [
          -0.4867810554948768,
          3.775693420481691
        ],
        "queries": 128,
        "parents": 128,
        "improved": 63,
        "harmed": 39,
        "unchanged": 26,
        "declined_over5pp": 22
      },
      "t": {
        "mean": 2.6059520321580467,
        "CI95": [
          0.6075779750504385,
          4.7571698937916205
        ],
        "queries": 128,
        "parents": 128,
        "improved": 58,
        "harmed": 40,
        "unchanged": 30,
        "declined_over5pp": 24
      },
      "observed_actual_sIoU": {
        "mean": 7.90622762597688,
        "CI95": [
          2.558660059588075,
          13.26621569104397
        ],
        "queries": 102,
        "parents": 102,
        "improved": 66,
        "harmed": 32,
        "unchanged": 4,
        "declined_over5pp": 24
      },
      "unobserved_actual_sIoU": {
        "mean": 2.294749027908817,
        "CI95": [
          -1.642546200864454,
          6.184964611897812
        ],
        "queries": 128,
        "parents": 128,
        "improved": 73,
        "harmed": 51,
        "unchanged": 4,
        "declined_over5pp": 31
      },
      "boundary_MAE_seconds": {
        "mean": -0.3102805278084092,
        "CI95": [
          -0.8906177581821816,
          0.19961884900769417
        ],
        "queries": 128,
        "parents": 128,
        "improved": 60,
        "harmed": 59,
        "unchanged": 9,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.6712796403809467,
        "CI95": [
          -1.8131395354014646,
          0.3911687709506319
        ],
        "queries": 128,
        "parents": 128,
        "improved": 37,
        "harmed": 38,
        "unchanged": 53,
        "declined_over5pp": 14
      },
      "t": {
        "mean": -0.5039608001998682,
        "CI95": [
          -2.313634088704681,
          1.1157942144523685
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 30,
        "unchanged": 68,
        "declined_over5pp": 15
      },
      "observed_actual_sIoU": {
        "mean": -0.7243372277217361,
        "CI95": [
          -2.6924097823434257,
          1.0061379263984225
        ],
        "queries": 100,
        "parents": 100,
        "improved": 29,
        "harmed": 34,
        "unchanged": 37,
        "declined_over5pp": 14
      },
      "unobserved_actual_sIoU": {
        "mean": -0.2642731411193458,
        "CI95": [
          -1.7031230480925734,
          1.156225823392515
        ],
        "queries": 128,
        "parents": 128,
        "improved": 41,
        "harmed": 46,
        "unchanged": 41,
        "declined_over5pp": 15
      },
      "boundary_MAE_seconds": {
        "mean": -0.12006337312640852,
        "CI95": [
          -0.49597462775236467,
          0.2630177277392378
        ],
        "queries": 128,
        "parents": 128,
        "improved": 37,
        "harmed": 37,
        "unchanged": 54,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.2760243743575175,
        "CI95": [
          -0.9818805763301901,
          0.3966025308803778
        ],
        "queries": 128,
        "parents": 128,
        "improved": 29,
        "harmed": 39,
        "unchanged": 60,
        "declined_over5pp": 6
      },
      "t": {
        "mean": -0.029012855576530994,
        "CI95": [
          -1.5883895935096883,
          1.6774147047626293
        ],
        "queries": 128,
        "parents": 128,
        "improved": 22,
        "harmed": 23,
        "unchanged": 83,
        "declined_over5pp": 13
      },
      "observed_actual_sIoU": {
        "mean": -0.693634828009443,
        "CI95": [
          -1.7773082564573564,
          0.2575508298822323
        ],
        "queries": 100,
        "parents": 100,
        "improved": 25,
        "harmed": 42,
        "unchanged": 33,
        "declined_over5pp": 6
      },
      "unobserved_actual_sIoU": {
        "mean": -0.764383694689551,
        "CI95": [
          -1.694498756325323,
          0.03334823507809104
        ],
        "queries": 128,
        "parents": 128,
        "improved": 42,
        "harmed": 52,
        "unchanged": 34,
        "declined_over5pp": 6
      },
      "boundary_MAE_seconds": {
        "mean": -0.22834986557588943,
        "CI95": [
          -0.7055590154124103,
          0.15104734122708913
        ],
        "queries": 128,
        "parents": 128,
        "improved": 33,
        "harmed": 30,
        "unchanged": 65,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 24,
      "severe_v": 22,
      "severe_union_once": 36,
      "rescue_union_once": 54
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 15,
      "severe_v": 14,
      "severe_union_once": 20,
      "rescue_union_once": 16
    },
    "Source-CleanSource": {
      "severe_t": 13,
      "severe_v": 6,
      "severe_union_once": 15,
      "rescue_union_once": 11
    }
  }
}
