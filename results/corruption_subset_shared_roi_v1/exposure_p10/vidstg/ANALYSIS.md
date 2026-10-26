# vidstg / exposure_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.12922352824370983,
      "t": 0.3132864163387188,
      "s": 0.35923419550246494,
      "fixed_actual_sIoU": 0.36345203411631644,
      "observed_actual_sIoU": 0.36276826626199277,
      "unobserved_actual_sIoU": 0.36423448386790824,
      "boundary_MAE": 222.3671875,
      "start_error_seconds": 7.408121846631438,
      "end_error_seconds": 8.262454160088122,
      "interval_length_seconds": 7.191479077605159,
      "length_error_seconds": 7.776549962078168,
      "vIoU_at_03": 0.140625,
      "vIoU_at_05": 0.046875,
      "vIoU_ge_03": 0.140625,
      "vIoU_ge_05": 0.046875,
      "tIoU_ge_05": 0.28125,
      "tIoU_eq_0": 0.234375,
      "boundary_MAE_seconds": 7.835288003359781
    },
    "FullMAP": {
      "v": 0.15136815059829412,
      "t": 0.3460985340770988,
      "s": 0.38787259193334434,
      "fixed_actual_sIoU": 0.3940358483459141,
      "observed_actual_sIoU": 0.4634353129855142,
      "unobserved_actual_sIoU": 0.3872346799850019,
      "boundary_MAE": 208.953125,
      "start_error_seconds": 7.3423553931299805,
      "end_error_seconds": 7.399856247571273,
      "interval_length_seconds": 10.778166157659756,
      "length_error_seconds": 8.38246946951665,
      "vIoU_at_03": 0.1796875,
      "vIoU_at_05": 0.1015625,
      "vIoU_ge_03": 0.1796875,
      "vIoU_ge_05": 0.1015625,
      "tIoU_ge_05": 0.328125,
      "tIoU_eq_0": 0.1796875,
      "boundary_MAE_seconds": 7.371105820350627
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
        "mean": 2.2144622354584262,
        "CI95": [
          0.04455713996359052,
          4.461739063309895
        ],
        "queries": 128,
        "parents": 128,
        "improved": 66,
        "harmed": 33,
        "unchanged": 29,
        "declined_over5pp": 20
      },
      "t": {
        "mean": 3.2812117738379922,
        "CI95": [
          0.9571566707256327,
          5.881299438371288
        ],
        "queries": 128,
        "parents": 128,
        "improved": 61,
        "harmed": 38,
        "unchanged": 29,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": 10.066704672352145,
        "CI95": [
          4.5848609726126694,
          15.531835948578662
        ],
        "queries": 103,
        "parents": 103,
        "improved": 71,
        "harmed": 28,
        "unchanged": 4,
        "declined_over5pp": 20
      },
      "unobserved_actual_sIoU": {
        "mean": 2.3000196117093714,
        "CI95": [
          -1.8616902879976227,
          6.4349827002965725
        ],
        "queries": 128,
        "parents": 128,
        "improved": 74,
        "harmed": 49,
        "unchanged": 5,
        "declined_over5pp": 33
      },
      "boundary_MAE_seconds": {
        "mean": -0.46418218300915387,
        "CI95": [
          -1.0962885202468367,
          0.08356363449619486
        ],
        "queries": 128,
        "parents": 128,
        "improved": 66,
        "harmed": 52,
        "unchanged": 10,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.20244398085245918,
        "CI95": [
          -1.0600503404052124,
          0.5517795570968912
        ],
        "queries": 128,
        "parents": 128,
        "improved": 38,
        "harmed": 34,
        "unchanged": 56,
        "declined_over5pp": 6
      },
      "t": {
        "mean": -0.5925813241306742,
        "CI95": [
          -1.9676641922138371,
          0.6393100897547568
        ],
        "queries": 128,
        "parents": 128,
        "improved": 21,
        "harmed": 26,
        "unchanged": 81,
        "declined_over5pp": 15
      },
      "observed_actual_sIoU": {
        "mean": 1.2563810601393886,
        "CI95": [
          0.017599865808868895,
          2.6336167476134196
        ],
        "queries": 101,
        "parents": 101,
        "improved": 29,
        "harmed": 36,
        "unchanged": 36,
        "declined_over5pp": 5
      },
      "unobserved_actual_sIoU": {
        "mean": 0.5158933418154033,
        "CI95": [
          -0.5118387208872899,
          1.6816817390813936
        ],
        "queries": 128,
        "parents": 128,
        "improved": 44,
        "harmed": 41,
        "unchanged": 43,
        "declined_over5pp": 8
      },
      "boundary_MAE_seconds": {
        "mean": 0.09975627388667041,
        "CI95": [
          -0.2533881201883258,
          0.461385029594506
        ],
        "queries": 128,
        "parents": 128,
        "improved": 27,
        "harmed": 34,
        "unchanged": 67,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.40054990097225746,
        "CI95": [
          -1.1945825329182271,
          0.28509449270349957
        ],
        "queries": 128,
        "parents": 128,
        "improved": 24,
        "harmed": 44,
        "unchanged": 60,
        "declined_over5pp": 4
      },
      "t": {
        "mean": -0.7928931211872827,
        "CI95": [
          -2.31646273689623,
          0.5324095114946821
        ],
        "queries": 128,
        "parents": 128,
        "improved": 20,
        "harmed": 22,
        "unchanged": 86,
        "declined_over5pp": 11
      },
      "observed_actual_sIoU": {
        "mean": -0.6420583101313886,
        "CI95": [
          -1.6034540622734068,
          0.07064344472062342
        ],
        "queries": 101,
        "parents": 101,
        "improved": 26,
        "harmed": 41,
        "unchanged": 34,
        "declined_over5pp": 5
      },
      "unobserved_actual_sIoU": {
        "mean": 0.010512204444642952,
        "CI95": [
          -0.3285123500512069,
          0.37880890596888533
        ],
        "queries": 128,
        "parents": 128,
        "improved": 45,
        "harmed": 49,
        "unchanged": 34,
        "declined_over5pp": 3
      },
      "boundary_MAE_seconds": {
        "mean": 0.14537143663793417,
        "CI95": [
          -0.08386089767990654,
          0.4083496664429598
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 28,
        "unchanged": 72,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 20,
      "severe_v": 20,
      "severe_union_once": 31,
      "rescue_union_once": 57
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 15,
      "severe_v": 6,
      "severe_union_once": 17,
      "rescue_union_once": 13
    },
    "Source-CleanSource": {
      "severe_t": 11,
      "severe_v": 4,
      "severe_union_once": 11,
      "rescue_union_once": 10
    }
  }
}
