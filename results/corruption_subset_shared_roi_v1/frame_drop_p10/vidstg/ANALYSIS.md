# vidstg / frame_drop_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.12608760186192836,
      "t": 0.31821806683426235,
      "s": 0.3482530187576639,
      "fixed_actual_sIoU": 0.35252844964641505,
      "observed_actual_sIoU": 0.3560097088410843,
      "unobserved_actual_sIoU": 0.3532658940490731,
      "boundary_MAE": 223.25390625,
      "start_error_seconds": 6.980616294998102,
      "end_error_seconds": 8.758359799759962,
      "interval_length_seconds": 7.346754801919233,
      "length_error_seconds": 7.680494072482453,
      "vIoU_at_03": 0.1484375,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.1484375,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.28125,
      "tIoU_eq_0": 0.25,
      "boundary_MAE_seconds": 7.869488047379031
    },
    "FullMAP": {
      "v": 0.13379323240113394,
      "t": 0.3352066478957929,
      "s": 0.3520988220354817,
      "fixed_actual_sIoU": 0.3588861408202435,
      "observed_actual_sIoU": 0.41830494356552794,
      "unobserved_actual_sIoU": 0.35349250919919717,
      "boundary_MAE": 215.76171875,
      "start_error_seconds": 7.057992578768325,
      "end_error_seconds": 8.261544381479203,
      "interval_length_seconds": 11.041669503146531,
      "length_error_seconds": 8.885158364226632,
      "vIoU_at_03": 0.1640625,
      "vIoU_at_05": 0.078125,
      "vIoU_ge_03": 0.1640625,
      "vIoU_ge_05": 0.078125,
      "tIoU_ge_05": 0.3046875,
      "tIoU_eq_0": 0.1875,
      "boundary_MAE_seconds": 7.6597684801237635
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
        "mean": 0.7705630539205561,
        "CI95": [
          -1.2447784880364974,
          2.740141743385209
        ],
        "queries": 128,
        "parents": 128,
        "improved": 62,
        "harmed": 37,
        "unchanged": 29,
        "declined_over5pp": 23
      },
      "t": {
        "mean": 1.6988581061530532,
        "CI95": [
          -0.30615717791028996,
          4.013761706822077
        ],
        "queries": 128,
        "parents": 128,
        "improved": 59,
        "harmed": 39,
        "unchanged": 30,
        "declined_over5pp": 27
      },
      "observed_actual_sIoU": {
        "mean": 6.229523472444369,
        "CI95": [
          0.862823418277735,
          11.56732203311586
        ],
        "queries": 105,
        "parents": 105,
        "improved": 65,
        "harmed": 33,
        "unchanged": 7,
        "declined_over5pp": 26
      },
      "unobserved_actual_sIoU": {
        "mean": 0.022661515012407873,
        "CI95": [
          -4.188615674580195,
          4.093756370730071
        ],
        "queries": 128,
        "parents": 128,
        "improved": 71,
        "harmed": 53,
        "unchanged": 4,
        "declined_over5pp": 36
      },
      "boundary_MAE_seconds": {
        "mean": -0.20971956725526736,
        "CI95": [
          -0.9001183505378123,
          0.4738883416337791
        ],
        "queries": 128,
        "parents": 128,
        "improved": 61,
        "harmed": 55,
        "unchanged": 12,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -1.9599358005684762,
        "CI95": [
          -3.794444853548662,
          -0.42135340882275524
        ],
        "queries": 128,
        "parents": 128,
        "improved": 33,
        "harmed": 42,
        "unchanged": 53,
        "declined_over5pp": 18
      },
      "t": {
        "mean": -1.681769942261256,
        "CI95": [
          -3.540951298378027,
          0.015469449437869461
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 39,
        "unchanged": 61,
        "declined_over5pp": 24
      },
      "observed_actual_sIoU": {
        "mean": -4.223558630082857,
        "CI95": [
          -7.772661226680096,
          -1.3223577027081048
        ],
        "queries": 100,
        "parents": 100,
        "improved": 27,
        "harmed": 40,
        "unchanged": 33,
        "declined_over5pp": 20
      },
      "unobserved_actual_sIoU": {
        "mean": -2.858323736765071,
        "CI95": [
          -5.686494859376996,
          -0.31609701408500407
        ],
        "queries": 128,
        "parents": 128,
        "improved": 33,
        "harmed": 54,
        "unchanged": 41,
        "declined_over5pp": 22
      },
      "boundary_MAE_seconds": {
        "mean": 0.38841893365980773,
        "CI95": [
          -0.15461340606385407,
          0.9997962853149385
        ],
        "queries": 128,
        "parents": 128,
        "improved": 35,
        "harmed": 45,
        "unchanged": 48,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.7141425391504046,
        "CI95": [
          -1.5453724824516935,
          0.0930684584456228
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 41,
        "unchanged": 59,
        "declined_over5pp": 12
      },
      "t": {
        "mean": -0.29972807163292514,
        "CI95": [
          -2.081141888460973,
          1.4777528274659477
        ],
        "queries": 128,
        "parents": 128,
        "improved": 26,
        "harmed": 34,
        "unchanged": 68,
        "declined_over5pp": 17
      },
      "observed_actual_sIoU": {
        "mean": -1.9360270670966424,
        "CI95": [
          -3.442334223675507,
          -0.689374846491506
        ],
        "queries": 100,
        "parents": 100,
        "improved": 21,
        "harmed": 45,
        "unchanged": 34,
        "declined_over5pp": 12
      },
      "unobserved_actual_sIoU": {
        "mean": -1.0863467774388673,
        "CI95": [
          -2.1226684238844733,
          -0.16038610603986597
        ],
        "queries": 128,
        "parents": 128,
        "improved": 43,
        "harmed": 52,
        "unchanged": 33,
        "declined_over5pp": 10
      },
      "boundary_MAE_seconds": {
        "mean": 0.17957148065718492,
        "CI95": [
          -0.26039601676841295,
          0.7644068507210582
        ],
        "queries": 128,
        "parents": 128,
        "improved": 42,
        "harmed": 36,
        "unchanged": 50,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 27,
      "severe_v": 23,
      "severe_union_once": 40,
      "rescue_union_once": 50
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 24,
      "severe_v": 18,
      "severe_union_once": 28,
      "rescue_union_once": 18
    },
    "Source-CleanSource": {
      "severe_t": 17,
      "severe_v": 12,
      "severe_union_once": 20,
      "rescue_union_once": 14
    }
  }
}
