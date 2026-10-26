# hc2 / frame_freeze_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.24794993582071226,
      "t": 0.46934258347837,
      "s": 0.5128088370394897,
      "fixed_actual_sIoU": 0.5161030148774938,
      "observed_actual_sIoU": 0.5089235529570043,
      "unobserved_actual_sIoU": 0.517332561874319,
      "boundary_MAE": 81.41015625,
      "start_error_seconds": 3.1423920571314823,
      "end_error_seconds": 3.012098174833234,
      "interval_length_seconds": 6.383214560213523,
      "length_error_seconds": 3.2128001713147505,
      "vIoU_at_03": 0.421875,
      "vIoU_at_05": 0.15625,
      "vIoU_ge_03": 0.421875,
      "vIoU_ge_05": 0.15625,
      "tIoU_ge_05": 0.484375,
      "tIoU_eq_0": 0.1328125,
      "boundary_MAE_seconds": 3.0772451159823584
    },
    "FullMAP": {
      "v": 0.31279996365192,
      "t": 0.5489596909462653,
      "s": 0.5381525029660836,
      "fixed_actual_sIoU": 0.5387520923471041,
      "observed_actual_sIoU": 0.5577459765251387,
      "unobserved_actual_sIoU": 0.534771308741922,
      "boundary_MAE": 75.38671875,
      "start_error_seconds": 2.8123413064011524,
      "end_error_seconds": 2.8949499973319694,
      "interval_length_seconds": 8.937507496709433,
      "length_error_seconds": 3.187451822000585,
      "vIoU_at_03": 0.5,
      "vIoU_at_05": 0.1875,
      "vIoU_ge_03": 0.5,
      "vIoU_ge_05": 0.1875,
      "tIoU_ge_05": 0.578125,
      "tIoU_eq_0": 0.09375,
      "boundary_MAE_seconds": 2.853645651866561
    },
    "CleanSource": {
      "v": 0.2512425286412039,
      "t": 0.46509477679431427,
      "s": 0.5256204548915578,
      "fixed_actual_sIoU": 0.5288800918917594,
      "observed_actual_sIoU": 0.52604723375025,
      "unobserved_actual_sIoU": 0.5306732451745338,
      "boundary_MAE": 83.53515625,
      "start_error_seconds": 3.1923334330505937,
      "end_error_seconds": 3.1303359777789206,
      "interval_length_seconds": 6.262376149251184,
      "length_error_seconds": 3.3165903062722952,
      "vIoU_at_03": 0.4296875,
      "vIoU_at_05": 0.140625,
      "vIoU_ge_03": 0.4296875,
      "vIoU_ge_05": 0.140625,
      "tIoU_ge_05": 0.453125,
      "tIoU_eq_0": 0.140625,
      "boundary_MAE_seconds": 3.1613347054147574
    },
    "CleanFullMAP": {
      "v": 0.31791298341569574,
      "t": 0.5496854831561468,
      "s": 0.5496602553288483,
      "fixed_actual_sIoU": 0.5519795310346984,
      "observed_actual_sIoU": 0.5799455876867262,
      "unobserved_actual_sIoU": 0.5463700370365687,
      "boundary_MAE": 76.87109375,
      "start_error_seconds": 2.78729433478442,
      "end_error_seconds": 3.0265915296068053,
      "interval_length_seconds": 8.876093507174193,
      "length_error_seconds": 3.198348835750862,
      "vIoU_at_03": 0.5078125,
      "vIoU_at_05": 0.2265625,
      "vIoU_ge_03": 0.5078125,
      "vIoU_ge_05": 0.2265625,
      "tIoU_ge_05": 0.5859375,
      "tIoU_eq_0": 0.09375,
      "boundary_MAE_seconds": 2.9069429321956126
    }
  },
  "paired": {
    "FullMAP-Source": {
      "v": {
        "mean": 6.485002783120775,
        "CI95": [
          4.00257555057348,
          9.008292917763807
        ],
        "queries": 128,
        "parents": 128,
        "improved": 86,
        "harmed": 30,
        "unchanged": 12,
        "declined_over5pp": 17
      },
      "t": {
        "mean": 7.961710746789523,
        "CI95": [
          4.989729601809083,
          10.934063550392354
        ],
        "queries": 128,
        "parents": 128,
        "improved": 81,
        "harmed": 32,
        "unchanged": 15,
        "declined_over5pp": 19
      },
      "observed_actual_sIoU": {
        "mean": 4.8822423568134115,
        "CI95": [
          0.9876476723102469,
          8.916866371119388
        ],
        "queries": 127,
        "parents": 127,
        "improved": 83,
        "harmed": 38,
        "unchanged": 6,
        "declined_over5pp": 26
      },
      "unobserved_actual_sIoU": {
        "mean": 1.7438746867603006,
        "CI95": [
          -0.7773795891491958,
          4.386437032836224
        ],
        "queries": 128,
        "parents": 128,
        "improved": 78,
        "harmed": 48,
        "unchanged": 2,
        "declined_over5pp": 23
      },
      "boundary_MAE_seconds": {
        "mean": -0.22359946411579756,
        "CI95": [
          -0.4139113286822127,
          -0.03678222127993841
        ],
        "queries": 128,
        "parents": 128,
        "improved": 76,
        "harmed": 46,
        "unchanged": 6,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.5113019763775712,
        "CI95": [
          -1.8025163362238696,
          0.8502037957377091
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 70,
        "unchanged": 12,
        "declined_over5pp": 21
      },
      "t": {
        "mean": -0.07257922098816583,
        "CI95": [
          -1.48206309097773,
          1.4132888153368044
        ],
        "queries": 128,
        "parents": 128,
        "improved": 25,
        "harmed": 32,
        "unchanged": 71,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": -1.7773055792340586,
        "CI95": [
          -4.358529629760272,
          0.7910791077443444
        ],
        "queries": 126,
        "parents": 126,
        "improved": 49,
        "harmed": 67,
        "unchanged": 10,
        "declined_over5pp": 29
      },
      "unobserved_actual_sIoU": {
        "mean": -1.1598728294646596,
        "CI95": [
          -2.9219247548460947,
          0.75782823947282
        ],
        "queries": 128,
        "parents": 128,
        "improved": 49,
        "harmed": 77,
        "unchanged": 2,
        "declined_over5pp": 23
      },
      "boundary_MAE_seconds": {
        "mean": -0.05329728032905164,
        "CI95": [
          -0.21578175100259622,
          0.09079070048930811
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 34,
        "unchanged": 66,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.32925928204916244,
        "CI95": [
          -1.621941844289297,
          1.1205001224444253
        ],
        "queries": 128,
        "parents": 128,
        "improved": 43,
        "harmed": 66,
        "unchanged": 19,
        "declined_over5pp": 23
      },
      "t": {
        "mean": 0.4247806684055714,
        "CI95": [
          -1.768805317154808,
          2.751896320409537
        ],
        "queries": 128,
        "parents": 128,
        "improved": 27,
        "harmed": 36,
        "unchanged": 65,
        "declined_over5pp": 25
      },
      "observed_actual_sIoU": {
        "mean": -1.3084604976126528,
        "CI95": [
          -2.9778306921560014,
          0.2900391104025216
        ],
        "queries": 126,
        "parents": 126,
        "improved": 48,
        "harmed": 70,
        "unchanged": 8,
        "declined_over5pp": 21
      },
      "unobserved_actual_sIoU": {
        "mean": -1.3340683300214824,
        "CI95": [
          -2.0907191528133175,
          -0.6237148797208432
        ],
        "queries": 128,
        "parents": 128,
        "improved": 48,
        "harmed": 76,
        "unchanged": 4,
        "declined_over5pp": 16
      },
      "boundary_MAE_seconds": {
        "mean": -0.08408958943239897,
        "CI95": [
          -0.3245241672446716,
          0.1153842848409762
        ],
        "queries": 128,
        "parents": 128,
        "improved": 27,
        "harmed": 41,
        "unchanged": 60,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 19,
      "severe_v": 17,
      "severe_union_once": 30,
      "rescue_union_once": 75
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 20,
      "severe_v": 21,
      "severe_union_once": 30,
      "rescue_union_once": 18
    },
    "Source-CleanSource": {
      "severe_t": 25,
      "severe_v": 23,
      "severe_union_once": 32,
      "rescue_union_once": 18
    }
  }
}
