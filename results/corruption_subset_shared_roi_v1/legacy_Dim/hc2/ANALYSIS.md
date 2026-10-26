# hc2 / legacy_Dim

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.2414394991213035,
      "t": 0.450934373272197,
      "s": 0.5209104858359563,
      "fixed_actual_sIoU": 0.5244455880047808,
      "observed_actual_sIoU": 0.5234230937905395,
      "unobserved_actual_sIoU": 0.5253886135682455,
      "boundary_MAE": 85.0703125,
      "start_error_seconds": 3.2638703413781,
      "end_error_seconds": 3.187304370215788,
      "interval_length_seconds": 6.548484648528094,
      "length_error_seconds": 3.534512307747134,
      "vIoU_at_03": 0.375,
      "vIoU_at_05": 0.125,
      "vIoU_ge_03": 0.375,
      "vIoU_ge_05": 0.125,
      "tIoU_ge_05": 0.46875,
      "tIoU_eq_0": 0.109375,
      "boundary_MAE_seconds": 3.225587355796944
    },
    "FullMAP": {
      "v": 0.3062699648645635,
      "t": 0.5441193053024631,
      "s": 0.534729208282825,
      "fixed_actual_sIoU": 0.5360285428541588,
      "observed_actual_sIoU": 0.5677896631334127,
      "unobserved_actual_sIoU": 0.530218210193964,
      "boundary_MAE": 76.296875,
      "start_error_seconds": 2.84840568657882,
      "end_error_seconds": 2.9256964305525823,
      "interval_length_seconds": 8.81713181565259,
      "length_error_seconds": 3.066803886659216,
      "vIoU_at_03": 0.46875,
      "vIoU_at_05": 0.21875,
      "vIoU_ge_03": 0.46875,
      "vIoU_ge_05": 0.21875,
      "tIoU_ge_05": 0.5390625,
      "tIoU_eq_0": 0.078125,
      "boundary_MAE_seconds": 2.887051058565701
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
        "mean": 6.483046574326002,
        "CI95": [
          3.836696208981135,
          9.118470560142121
        ],
        "queries": 128,
        "parents": 128,
        "improved": 83,
        "harmed": 35,
        "unchanged": 10,
        "declined_over5pp": 21
      },
      "t": {
        "mean": 9.318493203026605,
        "CI95": [
          6.073532316251233,
          12.766087845070308
        ],
        "queries": 128,
        "parents": 128,
        "improved": 86,
        "harmed": 30,
        "unchanged": 12,
        "declined_over5pp": 22
      },
      "observed_actual_sIoU": {
        "mean": 4.436656934287322,
        "CI95": [
          0.5454834972781665,
          8.20509929787868
        ],
        "queries": 126,
        "parents": 126,
        "improved": 85,
        "harmed": 35,
        "unchanged": 6,
        "declined_over5pp": 23
      },
      "unobserved_actual_sIoU": {
        "mean": 0.48295966257184075,
        "CI95": [
          -2.528338053315717,
          3.538638975664928
        ],
        "queries": 128,
        "parents": 128,
        "improved": 82,
        "harmed": 46,
        "unchanged": 0,
        "declined_over5pp": 27
      },
      "boundary_MAE_seconds": {
        "mean": -0.3385362972312428,
        "CI95": [
          -0.5579987895923798,
          -0.1278754660795474
        ],
        "queries": 128,
        "parents": 128,
        "improved": 73,
        "harmed": 50,
        "unchanged": 5,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -1.1643018551132205,
        "CI95": [
          -2.98922656666166,
          0.5094047107701676
        ],
        "queries": 128,
        "parents": 128,
        "improved": 61,
        "harmed": 57,
        "unchanged": 10,
        "declined_over5pp": 26
      },
      "t": {
        "mean": -0.5566177853683775,
        "CI95": [
          -2.936597895410245,
          1.7855707754821248
        ],
        "queries": 128,
        "parents": 128,
        "improved": 41,
        "harmed": 40,
        "unchanged": 47,
        "declined_over5pp": 28
      },
      "observed_actual_sIoU": {
        "mean": -1.2155924553313537,
        "CI95": [
          -4.7744501985605,
          2.318860024517337
        ],
        "queries": 126,
        "parents": 126,
        "improved": 57,
        "harmed": 67,
        "unchanged": 2,
        "declined_over5pp": 41
      },
      "unobserved_actual_sIoU": {
        "mean": -1.6151826842604715,
        "CI95": [
          -3.391248508179556,
          0.19919586903713546
        ],
        "queries": 128,
        "parents": 128,
        "improved": 53,
        "harmed": 75,
        "unchanged": 0,
        "declined_over5pp": 36
      },
      "boundary_MAE_seconds": {
        "mean": -0.01989187362991124,
        "CI95": [
          -0.24555149291635053,
          0.1871483860816678
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 41,
        "unchanged": 41,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.9803029519900375,
        "CI95": [
          -2.748976164798599,
          0.6801604150479454
        ],
        "queries": 128,
        "parents": 128,
        "improved": 48,
        "harmed": 66,
        "unchanged": 14,
        "declined_over5pp": 25
      },
      "t": {
        "mean": -1.4160403522117253,
        "CI95": [
          -4.18329258739353,
          1.192243147341299
        ],
        "queries": 128,
        "parents": 128,
        "improved": 44,
        "harmed": 50,
        "unchanged": 34,
        "declined_over5pp": 32
      },
      "observed_actual_sIoU": {
        "mean": -0.26241399597105414,
        "CI95": [
          -2.9073113348289366,
          2.355635667005403
        ],
        "queries": 126,
        "parents": 126,
        "improved": 58,
        "harmed": 62,
        "unchanged": 6,
        "declined_over5pp": 22
      },
      "unobserved_actual_sIoU": {
        "mean": -0.5284631606288349,
        "CI95": [
          -1.7542936845781039,
          0.7267727002503565
        ],
        "queries": 128,
        "parents": 128,
        "improved": 57,
        "harmed": 69,
        "unchanged": 2,
        "declined_over5pp": 16
      },
      "boundary_MAE_seconds": {
        "mean": 0.06425265038218661,
        "CI95": [
          -0.14176690552836854,
          0.2785217158718891
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 53,
        "unchanged": 29,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 22,
      "severe_v": 21,
      "severe_union_once": 35,
      "rescue_union_once": 78
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 28,
      "severe_v": 26,
      "severe_union_once": 38,
      "rescue_union_once": 31
    },
    "Source-CleanSource": {
      "severe_t": 32,
      "severe_v": 25,
      "severe_union_once": 36,
      "rescue_union_once": 29
    }
  }
}
