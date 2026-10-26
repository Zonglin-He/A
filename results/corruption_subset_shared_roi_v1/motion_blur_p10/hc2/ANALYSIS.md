# hc2 / motion_blur_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.23800134829359776,
      "t": 0.44360315058915845,
      "s": 0.5176957976246562,
      "fixed_actual_sIoU": 0.5214146341305035,
      "observed_actual_sIoU": 0.529162437511281,
      "unobserved_actual_sIoU": 0.5212102916940092,
      "boundary_MAE": 87.66015625,
      "start_error_seconds": 3.38310427106227,
      "end_error_seconds": 3.2424734881010417,
      "interval_length_seconds": 6.00214438215759,
      "length_error_seconds": 3.1932586641990657,
      "vIoU_at_03": 0.40625,
      "vIoU_at_05": 0.140625,
      "vIoU_ge_03": 0.40625,
      "vIoU_ge_05": 0.140625,
      "tIoU_ge_05": 0.4375,
      "tIoU_eq_0": 0.15625,
      "boundary_MAE_seconds": 3.3127888795816562
    },
    "FullMAP": {
      "v": 0.2940528661337778,
      "t": 0.5226301488606306,
      "s": 0.5392484708932762,
      "fixed_actual_sIoU": 0.5394552930560488,
      "observed_actual_sIoU": 0.569339857977716,
      "unobserved_actual_sIoU": 0.5348986716374806,
      "boundary_MAE": 82.8046875,
      "start_error_seconds": 3.1451274920708117,
      "end_error_seconds": 3.1136722164417687,
      "interval_length_seconds": 8.541450153763227,
      "length_error_seconds": 3.1649549467788747,
      "vIoU_at_03": 0.4375,
      "vIoU_at_05": 0.1953125,
      "vIoU_ge_03": 0.4375,
      "vIoU_ge_05": 0.1953125,
      "tIoU_ge_05": 0.53125,
      "tIoU_eq_0": 0.1015625,
      "boundary_MAE_seconds": 3.129399854256291
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
        "mean": 5.605151784018002,
        "CI95": [
          3.145731759327191,
          8.019840090917095
        ],
        "queries": 128,
        "parents": 128,
        "improved": 87,
        "harmed": 27,
        "unchanged": 14,
        "declined_over5pp": 16
      },
      "t": {
        "mean": 7.902699827147225,
        "CI95": [
          4.964751472663665,
          10.89353440702245
        ],
        "queries": 128,
        "parents": 128,
        "improved": 89,
        "harmed": 25,
        "unchanged": 14,
        "declined_over5pp": 19
      },
      "observed_actual_sIoU": {
        "mean": 4.017742046643503,
        "CI95": [
          0.09712540687230667,
          8.00569709053535
        ],
        "queries": 126,
        "parents": 126,
        "improved": 80,
        "harmed": 41,
        "unchanged": 5,
        "declined_over5pp": 30
      },
      "unobserved_actual_sIoU": {
        "mean": 1.3688379943471267,
        "CI95": [
          -1.3554758212617095,
          4.1507176100892424
        ],
        "queries": 128,
        "parents": 128,
        "improved": 74,
        "harmed": 52,
        "unchanged": 2,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": -0.18338902532536563,
        "CI95": [
          -0.35938756591316773,
          -0.0057135229207222055
        ],
        "queries": 128,
        "parents": 128,
        "improved": 84,
        "harmed": 40,
        "unchanged": 4,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -2.3860117281917943,
        "CI95": [
          -4.34684411572931,
          -0.4726094087881524
        ],
        "queries": 128,
        "parents": 128,
        "improved": 51,
        "harmed": 65,
        "unchanged": 12,
        "declined_over5pp": 27
      },
      "t": {
        "mean": -2.705533429551622,
        "CI95": [
          -5.241292566189762,
          -0.200444488448104
        ],
        "queries": 128,
        "parents": 128,
        "improved": 27,
        "harmed": 38,
        "unchanged": 63,
        "declined_over5pp": 29
      },
      "observed_actual_sIoU": {
        "mean": -1.0605729709010236,
        "CI95": [
          -4.519139479092536,
          2.4036282762630834
        ],
        "queries": 126,
        "parents": 126,
        "improved": 48,
        "harmed": 69,
        "unchanged": 9,
        "declined_over5pp": 28
      },
      "unobserved_actual_sIoU": {
        "mean": -1.1471365399088103,
        "CI95": [
          -3.1264373541280257,
          0.8293460659972255
        ],
        "queries": 128,
        "parents": 128,
        "improved": 52,
        "harmed": 74,
        "unchanged": 2,
        "declined_over5pp": 29
      },
      "boundary_MAE_seconds": {
        "mean": 0.22245692206067802,
        "CI95": [
          0.01882532590792076,
          0.4293977178045736
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 41,
        "unchanged": 57,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -1.3241180347606114,
        "CI95": [
          -3.11958839586569,
          0.5650195613449704
        ],
        "queries": 128,
        "parents": 128,
        "improved": 35,
        "harmed": 74,
        "unchanged": 19,
        "declined_over5pp": 26
      },
      "t": {
        "mean": -2.149162620515588,
        "CI95": [
          -4.903038396190514,
          0.6848725864519118
        ],
        "queries": 128,
        "parents": 128,
        "improved": 23,
        "harmed": 54,
        "unchanged": 51,
        "declined_over5pp": 36
      },
      "observed_actual_sIoU": {
        "mean": 0.31152037610309574,
        "CI95": [
          -2.0378139382988936,
          2.704366360556925
        ],
        "queries": 126,
        "parents": 126,
        "improved": 60,
        "harmed": 59,
        "unchanged": 7,
        "declined_over5pp": 18
      },
      "unobserved_actual_sIoU": {
        "mean": -0.9462953480524599,
        "CI95": [
          -1.8594509130632353,
          0.0030904992110918454
        ],
        "queries": 128,
        "parents": 128,
        "improved": 40,
        "harmed": 84,
        "unchanged": 4,
        "declined_over5pp": 11
      },
      "boundary_MAE_seconds": {
        "mean": 0.15145417416689871,
        "CI95": [
          -0.07710139831394833,
          0.37866218474199004
        ],
        "queries": 128,
        "parents": 128,
        "improved": 27,
        "harmed": 57,
        "unchanged": 44,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 19,
      "severe_v": 16,
      "severe_union_once": 26,
      "rescue_union_once": 78
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 29,
      "severe_v": 27,
      "severe_union_once": 37,
      "rescue_union_once": 23
    },
    "Source-CleanSource": {
      "severe_t": 36,
      "severe_v": 26,
      "severe_union_once": 37,
      "rescue_union_once": 15
    }
  }
}
