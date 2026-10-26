# hc2 / frame_drop_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.20896284393285258,
      "t": 0.39516104981707745,
      "s": 0.5068211508012648,
      "fixed_actual_sIoU": 0.5092016851328982,
      "observed_actual_sIoU": 0.5241642038418661,
      "unobserved_actual_sIoU": 0.5086239300249631,
      "boundary_MAE": 96.734375,
      "start_error_seconds": 3.6258117057793817,
      "end_error_seconds": 3.6753909169649766,
      "interval_length_seconds": 5.102062525533114,
      "length_error_seconds": 3.8040493735375636,
      "vIoU_at_03": 0.359375,
      "vIoU_at_05": 0.078125,
      "vIoU_ge_03": 0.359375,
      "vIoU_ge_05": 0.078125,
      "tIoU_ge_05": 0.3671875,
      "tIoU_eq_0": 0.1796875,
      "boundary_MAE_seconds": 3.6506013113721787
    },
    "FullMAP": {
      "v": 0.27722651330335596,
      "t": 0.4972142635813358,
      "s": 0.5196824906401774,
      "fixed_actual_sIoU": 0.5190125521990057,
      "observed_actual_sIoU": 0.5649773936745627,
      "unobserved_actual_sIoU": 0.5119342290790824,
      "boundary_MAE": 87.34375,
      "start_error_seconds": 3.312173619455769,
      "end_error_seconds": 3.266700660710832,
      "interval_length_seconds": 7.89570988206858,
      "length_error_seconds": 3.2355595174178378,
      "vIoU_at_03": 0.453125,
      "vIoU_at_05": 0.171875,
      "vIoU_ge_03": 0.453125,
      "vIoU_ge_05": 0.171875,
      "tIoU_ge_05": 0.4921875,
      "tIoU_eq_0": 0.1015625,
      "boundary_MAE_seconds": 3.289437140083301
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
        "mean": 6.826366937050339,
        "CI95": [
          4.4016629016803845,
          9.201313334692193
        ],
        "queries": 128,
        "parents": 128,
        "improved": 87,
        "harmed": 25,
        "unchanged": 16,
        "declined_over5pp": 15
      },
      "t": {
        "mean": 10.205321376425847,
        "CI95": [
          7.154812303154631,
          13.218051370999023
        ],
        "queries": 128,
        "parents": 128,
        "improved": 87,
        "harmed": 23,
        "unchanged": 18,
        "declined_over5pp": 14
      },
      "observed_actual_sIoU": {
        "mean": 4.081318983269666,
        "CI95": [
          0.6825617802758858,
          7.5505557750488155
        ],
        "queries": 126,
        "parents": 126,
        "improved": 83,
        "harmed": 38,
        "unchanged": 5,
        "declined_over5pp": 26
      },
      "unobserved_actual_sIoU": {
        "mean": 0.33102990541193245,
        "CI95": [
          -2.2890120601196453,
          2.9425991085378715
        ],
        "queries": 128,
        "parents": 128,
        "improved": 70,
        "harmed": 57,
        "unchanged": 1,
        "declined_over5pp": 28
      },
      "boundary_MAE_seconds": {
        "mean": -0.36116417128887823,
        "CI95": [
          -0.5721690762512188,
          -0.1541446103691778
        ],
        "queries": 128,
        "parents": 128,
        "improved": 79,
        "harmed": 42,
        "unchanged": 7,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -4.068647011233974,
        "CI95": [
          -6.614921269346533,
          -1.6130697618215308
        ],
        "queries": 128,
        "parents": 128,
        "improved": 45,
        "harmed": 72,
        "unchanged": 11,
        "declined_over5pp": 40
      },
      "t": {
        "mean": -5.247121957481095,
        "CI95": [
          -8.537140824768073,
          -1.8749740513847477
        ],
        "queries": 128,
        "parents": 128,
        "improved": 34,
        "harmed": 56,
        "unchanged": 38,
        "declined_over5pp": 47
      },
      "observed_actual_sIoU": {
        "mean": -1.6688496459019262,
        "CI95": [
          -5.248472808443565,
          1.8567816568991469
        ],
        "queries": 125,
        "parents": 125,
        "improved": 54,
        "harmed": 63,
        "unchanged": 8,
        "declined_over5pp": 36
      },
      "unobserved_actual_sIoU": {
        "mean": -3.4435807957486255,
        "CI95": [
          -5.540229667420511,
          -1.4925580870459616
        ],
        "queries": 128,
        "parents": 128,
        "improved": 51,
        "harmed": 76,
        "unchanged": 1,
        "declined_over5pp": 36
      },
      "boundary_MAE_seconds": {
        "mean": 0.38249420788768834,
        "CI95": [
          0.11555924017441348,
          0.6554028443651359
        ],
        "queries": 128,
        "parents": 128,
        "improved": 35,
        "harmed": 58,
        "unchanged": 35,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -4.227968470835129,
        "CI95": [
          -6.10569756284155,
          -2.338990722325824
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 79,
        "unchanged": 19,
        "declined_over5pp": 43
      },
      "t": {
        "mean": -6.993372697723682,
        "CI95": [
          -10.371357521882578,
          -3.7118766623931023
        ],
        "queries": 128,
        "parents": 128,
        "improved": 24,
        "harmed": 64,
        "unchanged": 40,
        "declined_over5pp": 56
      },
      "observed_actual_sIoU": {
        "mean": -0.66882275790526,
        "CI95": [
          -2.526641571264241,
          1.334270710230016
        ],
        "queries": 125,
        "parents": 125,
        "improved": 54,
        "harmed": 66,
        "unchanged": 5,
        "declined_over5pp": 26
      },
      "unobserved_actual_sIoU": {
        "mean": -2.204931514957081,
        "CI95": [
          -3.187624050657888,
          -1.3435672753884742
        ],
        "queries": 128,
        "parents": 128,
        "improved": 32,
        "harmed": 93,
        "unchanged": 3,
        "declined_over5pp": 19
      },
      "boundary_MAE_seconds": {
        "mean": 0.4892666059574216,
        "CI95": [
          0.19713801387842225,
          0.7749552522022565
        ],
        "queries": 128,
        "parents": 128,
        "improved": 29,
        "harmed": 67,
        "unchanged": 32,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 14,
      "severe_v": 15,
      "severe_union_once": 23,
      "rescue_union_once": 78
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 47,
      "severe_v": 40,
      "severe_union_once": 52,
      "rescue_union_once": 28
    },
    "Source-CleanSource": {
      "severe_t": 56,
      "severe_v": 43,
      "severe_union_once": 60,
      "rescue_union_once": 15
    }
  }
}
