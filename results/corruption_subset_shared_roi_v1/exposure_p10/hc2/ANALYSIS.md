# hc2 / exposure_p10

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.24106094436470174,
      "t": 0.4478651408273531,
      "s": 0.5217614303143568,
      "fixed_actual_sIoU": 0.5268048714421452,
      "observed_actual_sIoU": 0.5256203790588081,
      "unobserved_actual_sIoU": 0.5291909310281774,
      "boundary_MAE": 85.94140625,
      "start_error_seconds": 3.347337342756253,
      "end_error_seconds": 3.1707652792867393,
      "interval_length_seconds": 6.013854586455005,
      "length_error_seconds": 3.352764964328908,
      "vIoU_at_03": 0.4375,
      "vIoU_at_05": 0.1328125,
      "vIoU_ge_03": 0.4375,
      "vIoU_ge_05": 0.1328125,
      "tIoU_ge_05": 0.46875,
      "tIoU_eq_0": 0.1484375,
      "boundary_MAE_seconds": 3.259051311021497
    },
    "FullMAP": {
      "v": 0.30711304442980775,
      "t": 0.5331227273986745,
      "s": 0.5501877601736762,
      "fixed_actual_sIoU": 0.5527435252062749,
      "observed_actual_sIoU": 0.5779532238999613,
      "unobserved_actual_sIoU": 0.5486722459722875,
      "boundary_MAE": 79.62890625,
      "start_error_seconds": 2.981986312414181,
      "end_error_seconds": 3.0352416357615692,
      "interval_length_seconds": 8.767351539323727,
      "length_error_seconds": 3.2504338663319396,
      "vIoU_at_03": 0.4921875,
      "vIoU_at_05": 0.2109375,
      "vIoU_ge_03": 0.4921875,
      "vIoU_ge_05": 0.2109375,
      "tIoU_ge_05": 0.546875,
      "tIoU_eq_0": 0.09375,
      "boundary_MAE_seconds": 3.0086139740878752
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
        "mean": 6.605210006510599,
        "CI95": [
          3.8991768284909716,
          9.318327110601917
        ],
        "queries": 128,
        "parents": 128,
        "improved": 84,
        "harmed": 30,
        "unchanged": 14,
        "declined_over5pp": 19
      },
      "t": {
        "mean": 8.525758657132132,
        "CI95": [
          5.4867291809199195,
          11.550495287891483
        ],
        "queries": 128,
        "parents": 128,
        "improved": 84,
        "harmed": 30,
        "unchanged": 14,
        "declined_over5pp": 17
      },
      "observed_actual_sIoU": {
        "mean": 5.233284484115325,
        "CI95": [
          0.9254363881501182,
          9.582237619408604
        ],
        "queries": 126,
        "parents": 126,
        "improved": 82,
        "harmed": 40,
        "unchanged": 4,
        "declined_over5pp": 31
      },
      "unobserved_actual_sIoU": {
        "mean": 1.9481314944110237,
        "CI95": [
          -0.7308346769344937,
          4.68583652773964
        ],
        "queries": 128,
        "parents": 128,
        "improved": 83,
        "harmed": 43,
        "unchanged": 2,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": -0.25043733693362147,
        "CI95": [
          -0.43325563618231283,
          -0.0654076450822583
        ],
        "queries": 128,
        "parents": 128,
        "improved": 78,
        "harmed": 47,
        "unchanged": 3,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -1.0799938985887994,
        "CI95": [
          -2.6534348169158193,
          0.3927487516321825
        ],
        "queries": 128,
        "parents": 128,
        "improved": 56,
        "harmed": 59,
        "unchanged": 13,
        "declined_over5pp": 19
      },
      "t": {
        "mean": -1.6562755757472414,
        "CI95": [
          -3.4697582244885923,
          -0.05526087191944698
        ],
        "queries": 128,
        "parents": 128,
        "improved": 25,
        "harmed": 28,
        "unchanged": 75,
        "declined_over5pp": 24
      },
      "observed_actual_sIoU": {
        "mean": -0.19923637867650631,
        "CI95": [
          -1.7973497197997057,
          1.623166582778623
        ],
        "queries": 126,
        "parents": 126,
        "improved": 50,
        "harmed": 63,
        "unchanged": 13,
        "declined_over5pp": 21
      },
      "unobserved_actual_sIoU": {
        "mean": 0.23022089357189376,
        "CI95": [
          -0.8260035052548867,
          1.3825714229122623
        ],
        "queries": 128,
        "parents": 128,
        "improved": 59,
        "harmed": 67,
        "unchanged": 2,
        "declined_over5pp": 13
      },
      "boundary_MAE_seconds": {
        "mean": 0.10167104189226261,
        "CI95": [
          -0.04333380020232489,
          0.25528579428601195
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 31,
        "unchanged": 69,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -1.0181584276502142,
        "CI95": [
          -2.0854590273842795,
          0.007988560136932967
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 61,
        "unchanged": 21,
        "declined_over5pp": 13
      },
      "t": {
        "mean": -1.7229635966961132,
        "CI95": [
          -3.514297969391426,
          -0.07516503899773222
        ],
        "queries": 128,
        "parents": 128,
        "improved": 24,
        "harmed": 38,
        "unchanged": 66,
        "declined_over5pp": 19
      },
      "observed_actual_sIoU": {
        "mean": -0.04268546914420884,
        "CI95": [
          -1.3111430749263295,
          1.4301699363948153
        ],
        "queries": 126,
        "parents": 126,
        "improved": 58,
        "harmed": 60,
        "unchanged": 8,
        "declined_over5pp": 14
      },
      "unobserved_actual_sIoU": {
        "mean": -0.14823141463565215,
        "CI95": [
          -0.6860783610416697,
          0.4614696823468899
        ],
        "queries": 128,
        "parents": 128,
        "improved": 53,
        "harmed": 72,
        "unchanged": 3,
        "declined_over5pp": 4
      },
      "boundary_MAE_seconds": {
        "mean": 0.09771660560673921,
        "CI95": [
          -0.0018950550840030251,
          0.21304674966426637
        ],
        "queries": 128,
        "parents": 128,
        "improved": 29,
        "harmed": 42,
        "unchanged": 57,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 17,
      "severe_v": 19,
      "severe_union_once": 30,
      "rescue_union_once": 78
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 24,
      "severe_v": 19,
      "severe_union_once": 27,
      "rescue_union_once": 21
    },
    "Source-CleanSource": {
      "severe_t": 19,
      "severe_v": 13,
      "severe_union_once": 19,
      "rescue_union_once": 16
    }
  }
}
