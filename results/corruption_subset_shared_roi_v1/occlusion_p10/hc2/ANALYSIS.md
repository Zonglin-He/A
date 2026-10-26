# hc2 / occlusion_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.23269773869097182,
      "t": 0.43773041286704883,
      "s": 0.5157074458901657,
      "fixed_actual_sIoU": 0.51894986061319,
      "observed_actual_sIoU": 0.530052187475023,
      "unobserved_actual_sIoU": 0.5187072630116188,
      "boundary_MAE": 87.84765625,
      "start_error_seconds": 3.2829694245318057,
      "end_error_seconds": 3.3511030682001577,
      "interval_length_seconds": 5.847333883367584,
      "length_error_seconds": 3.3932485046109546,
      "vIoU_at_03": 0.40625,
      "vIoU_at_05": 0.1328125,
      "vIoU_ge_03": 0.40625,
      "vIoU_ge_05": 0.1328125,
      "tIoU_ge_05": 0.4296875,
      "tIoU_eq_0": 0.1484375,
      "boundary_MAE_seconds": 3.3170362463659817
    },
    "FullMAP": {
      "v": 0.3046894128124248,
      "t": 0.5280616110445036,
      "s": 0.5494196312560657,
      "fixed_actual_sIoU": 0.5511383829539569,
      "observed_actual_sIoU": 0.5903401576547557,
      "unobserved_actual_sIoU": 0.5445482613274729,
      "boundary_MAE": 81.0390625,
      "start_error_seconds": 3.043333063651837,
      "end_error_seconds": 3.0757149638296886,
      "interval_length_seconds": 8.46293522997826,
      "length_error_seconds": 3.145788516315735,
      "vIoU_at_03": 0.5078125,
      "vIoU_at_05": 0.1875,
      "vIoU_ge_03": 0.5078125,
      "vIoU_ge_05": 0.1875,
      "tIoU_ge_05": 0.5234375,
      "tIoU_eq_0": 0.1015625,
      "boundary_MAE_seconds": 3.059524013740763
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
        "mean": 7.199167412145299,
        "CI95": [
          4.620837398080746,
          9.755676466409392
        ],
        "queries": 128,
        "parents": 128,
        "improved": 88,
        "harmed": 26,
        "unchanged": 14,
        "declined_over5pp": 16
      },
      "t": {
        "mean": 9.033119817745467,
        "CI95": [
          6.085954774063677,
          12.081836954464901
        ],
        "queries": 128,
        "parents": 128,
        "improved": 88,
        "harmed": 24,
        "unchanged": 16,
        "declined_over5pp": 16
      },
      "observed_actual_sIoU": {
        "mean": 6.028797017973269,
        "CI95": [
          2.083868340283584,
          10.061906166615078
        ],
        "queries": 126,
        "parents": 126,
        "improved": 82,
        "harmed": 39,
        "unchanged": 5,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 2.584099831585402,
        "CI95": [
          -0.25600832405914187,
          5.532589750820882
        ],
        "queries": 128,
        "parents": 128,
        "improved": 75,
        "harmed": 53,
        "unchanged": 0,
        "declined_over5pp": 29
      },
      "boundary_MAE_seconds": {
        "mean": -0.2575122326252192,
        "CI95": [
          -0.4164594164981464,
          -0.09977731395627623
        ],
        "queries": 128,
        "parents": 128,
        "improved": 86,
        "harmed": 36,
        "unchanged": 6,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -1.3223570603270909,
        "CI95": [
          -2.98527962000049,
          0.26748459155325255
        ],
        "queries": 128,
        "parents": 128,
        "improved": 54,
        "harmed": 63,
        "unchanged": 11,
        "declined_over5pp": 24
      },
      "t": {
        "mean": -2.1623872111643334,
        "CI95": [
          -4.394067728556746,
          -0.038095385163859415
        ],
        "queries": 128,
        "parents": 128,
        "improved": 26,
        "harmed": 32,
        "unchanged": 70,
        "declined_over5pp": 27
      },
      "observed_actual_sIoU": {
        "mean": 1.0394569968029412,
        "CI95": [
          -1.686841417030671,
          3.8486514483039937
        ],
        "queries": 126,
        "parents": 126,
        "improved": 58,
        "harmed": 56,
        "unchanged": 12,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": -0.1821775709095768,
        "CI95": [
          -1.7619579831798522,
          1.5116343005453312
        ],
        "queries": 128,
        "parents": 128,
        "improved": 55,
        "harmed": 73,
        "unchanged": 0,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": 0.15258108154515024,
        "CI95": [
          -0.03156166617050581,
          0.33679273408695226
        ],
        "queries": 128,
        "parents": 128,
        "improved": 29,
        "harmed": 36,
        "unchanged": 63,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -1.8544789950232046,
        "CI95": [
          -3.350408046914517,
          -0.33152454656925934
        ],
        "queries": 128,
        "parents": 128,
        "improved": 34,
        "harmed": 75,
        "unchanged": 19,
        "declined_over5pp": 26
      },
      "t": {
        "mean": -2.736436392726542,
        "CI95": [
          -4.945546295108246,
          -0.4908458228959393
        ],
        "queries": 128,
        "parents": 128,
        "improved": 17,
        "harmed": 45,
        "unchanged": 66,
        "declined_over5pp": 34
      },
      "observed_actual_sIoU": {
        "mean": 0.4004953724772949,
        "CI95": [
          -1.4480088693913282,
          2.309337405759729
        ],
        "queries": 126,
        "parents": 126,
        "improved": 52,
        "harmed": 65,
        "unchanged": 9,
        "declined_over5pp": 16
      },
      "unobserved_actual_sIoU": {
        "mean": -1.1965982162915012,
        "CI95": [
          -2.144325378606923,
          -0.276574365109327
        ],
        "queries": 128,
        "parents": 128,
        "improved": 49,
        "harmed": 76,
        "unchanged": 3,
        "declined_over5pp": 15
      },
      "boundary_MAE_seconds": {
        "mean": 0.15570154095122457,
        "CI95": [
          -0.024040118982345086,
          0.3313367719674612
        ],
        "queries": 128,
        "parents": 128,
        "improved": 21,
        "harmed": 47,
        "unchanged": 60,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 16,
      "severe_v": 16,
      "severe_union_once": 27,
      "rescue_union_once": 77
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 27,
      "severe_v": 24,
      "severe_union_once": 32,
      "rescue_union_once": 26
    },
    "Source-CleanSource": {
      "severe_t": 34,
      "severe_v": 26,
      "severe_union_once": 35,
      "rescue_union_once": 15
    }
  }
}
