# vidstg / occlusion_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.1328668157561319,
      "t": 0.32790792897719545,
      "s": 0.35511538328904696,
      "fixed_actual_sIoU": 0.3593782004998627,
      "observed_actual_sIoU": 0.3611645037358441,
      "unobserved_actual_sIoU": 0.3585939813707113,
      "boundary_MAE": 209.11328125,
      "start_error_seconds": 6.706284381040849,
      "end_error_seconds": 8.059739978328238,
      "interval_length_seconds": 7.905670076025453,
      "length_error_seconds": 7.455113654286001,
      "vIoU_at_03": 0.15625,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.15625,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.3046875,
      "tIoU_eq_0": 0.21875,
      "boundary_MAE_seconds": 7.383012179684544
    },
    "FullMAP": {
      "v": 0.15233764767636415,
      "t": 0.35394743578826904,
      "s": 0.3804035824564691,
      "fixed_actual_sIoU": 0.385242713681287,
      "observed_actual_sIoU": 0.4414893367130022,
      "unobserved_actual_sIoU": 0.3792020435548082,
      "boundary_MAE": 204.77734375,
      "start_error_seconds": 6.813816145828655,
      "end_error_seconds": 7.6929992435399805,
      "interval_length_seconds": 11.06195396365608,
      "length_error_seconds": 8.04529248276111,
      "vIoU_at_03": 0.1875,
      "vIoU_at_05": 0.109375,
      "vIoU_ge_03": 0.1875,
      "vIoU_ge_05": 0.109375,
      "tIoU_ge_05": 0.3359375,
      "tIoU_eq_0": 0.171875,
      "boundary_MAE_seconds": 7.253407694684318
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
        "mean": 1.947083192023224,
        "CI95": [
          -0.2203150420556499,
          4.138166488203015
        ],
        "queries": 128,
        "parents": 128,
        "improved": 66,
        "harmed": 37,
        "unchanged": 25,
        "declined_over5pp": 22
      },
      "t": {
        "mean": 2.6039506811073556,
        "CI95": [
          0.5021592083247274,
          4.781665113165364
        ],
        "queries": 128,
        "parents": 128,
        "improved": 57,
        "harmed": 40,
        "unchanged": 31,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 8.032483297715808,
        "CI95": [
          2.6253736046137175,
          13.48932312001785
        ],
        "queries": 101,
        "parents": 101,
        "improved": 67,
        "harmed": 30,
        "unchanged": 4,
        "declined_over5pp": 26
      },
      "unobserved_actual_sIoU": {
        "mean": 2.060806218409689,
        "CI95": [
          -1.947214105847524,
          6.042269880232617
        ],
        "queries": 128,
        "parents": 128,
        "improved": 71,
        "harmed": 53,
        "unchanged": 4,
        "declined_over5pp": 32
      },
      "boundary_MAE_seconds": {
        "mean": -0.12960448500022495,
        "CI95": [
          -0.7179602336657687,
          0.42598738190001484
        ],
        "queries": 128,
        "parents": 128,
        "improved": 61,
        "harmed": 56,
        "unchanged": 11,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.10549427304545506,
        "CI95": [
          -0.6992774289343535,
          0.46632111122353853
        ],
        "queries": 128,
        "parents": 128,
        "improved": 37,
        "harmed": 34,
        "unchanged": 57,
        "declined_over5pp": 6
      },
      "t": {
        "mean": 0.19230884698635414,
        "CI95": [
          -0.9670848695156637,
          1.3309720606469564
        ],
        "queries": 128,
        "parents": 128,
        "improved": 25,
        "harmed": 26,
        "unchanged": 77,
        "declined_over5pp": 9
      },
      "observed_actual_sIoU": {
        "mean": -0.20177876684229779,
        "CI95": [
          -1.384662897438472,
          1.0986313049244407
        ],
        "queries": 101,
        "parents": 101,
        "improved": 24,
        "harmed": 38,
        "unchanged": 39,
        "declined_over5pp": 7
      },
      "unobserved_actual_sIoU": {
        "mean": -0.2873703012039705,
        "CI95": [
          -1.0988264463589137,
          0.6056135973395552
        ],
        "queries": 128,
        "parents": 128,
        "improved": 34,
        "harmed": 50,
        "unchanged": 44,
        "declined_over5pp": 11
      },
      "boundary_MAE_seconds": {
        "mean": -0.017941851779638404,
        "CI95": [
          -0.31547705781061547,
          0.27895260678082257
        ],
        "queries": 128,
        "parents": 128,
        "improved": 32,
        "harmed": 32,
        "unchanged": 64,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.03622114973005116,
        "CI95": [
          -0.42969496049022055,
          0.32159240060637706
        ],
        "queries": 128,
        "parents": 128,
        "improved": 32,
        "harmed": 34,
        "unchanged": 62,
        "declined_over5pp": 2
      },
      "t": {
        "mean": 0.6692581426603822,
        "CI95": [
          -0.49397309396298844,
          2.0875008532674446
        ],
        "queries": 128,
        "parents": 128,
        "improved": 18,
        "harmed": 17,
        "unchanged": 93,
        "declined_over5pp": 4
      },
      "observed_actual_sIoU": {
        "mean": -0.6933035592785516,
        "CI95": [
          -1.6027812925235998,
          -0.026950034955032863
        ],
        "queries": 101,
        "parents": 101,
        "improved": 30,
        "harmed": 36,
        "unchanged": 35,
        "declined_over5pp": 3
      },
      "unobserved_actual_sIoU": {
        "mean": -0.5535380452750482,
        "CI95": [
          -1.1811121817531824,
          -0.0639853360939036
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 46,
        "unchanged": 36,
        "declined_over5pp": 4
      },
      "boundary_MAE_seconds": {
        "mean": -0.30690438703730355,
        "CI95": [
          -0.7512565713377899,
          0.01817432020245102
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 22,
        "unchanged": 78,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 23,
      "severe_v": 22,
      "severe_union_once": 35,
      "rescue_union_once": 56
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 9,
      "severe_v": 6,
      "severe_union_once": 11,
      "rescue_union_once": 10
    },
    "Source-CleanSource": {
      "severe_t": 4,
      "severe_v": 2,
      "severe_union_once": 5,
      "rescue_union_once": 10
    }
  }
}
