# hc2 / motion_blur_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.24362484129690606,
      "t": 0.4537033754732762,
      "s": 0.522336821205666,
      "fixed_actual_sIoU": 0.5260709106277475,
      "observed_actual_sIoU": 0.5253626033756874,
      "unobserved_actual_sIoU": 0.5272724798131057,
      "boundary_MAE": 86.1796875,
      "start_error_seconds": 3.3022609320823673,
      "end_error_seconds": 3.217034315200932,
      "interval_length_seconds": 5.994428065460824,
      "length_error_seconds": 3.3048558615673187,
      "vIoU_at_03": 0.40625,
      "vIoU_at_05": 0.1328125,
      "vIoU_ge_03": 0.40625,
      "vIoU_ge_05": 0.1328125,
      "tIoU_ge_05": 0.4609375,
      "tIoU_eq_0": 0.1484375,
      "boundary_MAE_seconds": 3.25964762364165
    },
    "FullMAP": {
      "v": 0.30264322901131513,
      "t": 0.5262169362397273,
      "s": 0.5393752071517838,
      "fixed_actual_sIoU": 0.540891502370367,
      "observed_actual_sIoU": 0.5690272635817152,
      "unobserved_actual_sIoU": 0.5361273326015976,
      "boundary_MAE": 83.05859375,
      "start_error_seconds": 3.0804645655323664,
      "end_error_seconds": 3.2042405037440362,
      "interval_length_seconds": 8.688438549475638,
      "length_error_seconds": 3.2412267206192054,
      "vIoU_at_03": 0.484375,
      "vIoU_at_05": 0.203125,
      "vIoU_ge_03": 0.484375,
      "vIoU_ge_05": 0.203125,
      "tIoU_ge_05": 0.5390625,
      "tIoU_eq_0": 0.1015625,
      "boundary_MAE_seconds": 3.1423525346382015
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
        "mean": 5.901838771440909,
        "CI95": [
          3.455595876929047,
          8.375312129488211
        ],
        "queries": 128,
        "parents": 128,
        "improved": 77,
        "harmed": 37,
        "unchanged": 14,
        "declined_over5pp": 18
      },
      "t": {
        "mean": 7.251356076645127,
        "CI95": [
          4.291758755700059,
          10.391224915553819
        ],
        "queries": 128,
        "parents": 128,
        "improved": 77,
        "harmed": 34,
        "unchanged": 17,
        "declined_over5pp": 26
      },
      "observed_actual_sIoU": {
        "mean": 4.366466020602788,
        "CI95": [
          0.180330355819487,
          8.491768467419318
        ],
        "queries": 126,
        "parents": 126,
        "improved": 84,
        "harmed": 38,
        "unchanged": 4,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 0.8854852788491963,
        "CI95": [
          -1.811639255456993,
          3.552717048689239
        ],
        "queries": 128,
        "parents": 128,
        "improved": 75,
        "harmed": 51,
        "unchanged": 2,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": -0.1172950890034486,
        "CI95": [
          -0.30301070528033813,
          0.06365715423434903
        ],
        "queries": 128,
        "parents": 128,
        "improved": 71,
        "harmed": 50,
        "unchanged": 7,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -1.526975440438054,
        "CI95": [
          -2.6867491602923392,
          -0.5134589457862606
        ],
        "queries": 128,
        "parents": 128,
        "improved": 47,
        "harmed": 67,
        "unchanged": 14,
        "declined_over5pp": 18
      },
      "t": {
        "mean": -2.3468546916419433,
        "CI95": [
          -3.943834006158643,
          -0.948756128907646
        ],
        "queries": 128,
        "parents": 128,
        "improved": 15,
        "harmed": 27,
        "unchanged": 86,
        "declined_over5pp": 19
      },
      "observed_actual_sIoU": {
        "mean": -1.09183241050111,
        "CI95": [
          -3.593220087304031,
          1.212447497617401
        ],
        "queries": 126,
        "parents": 126,
        "improved": 53,
        "harmed": 58,
        "unchanged": 15,
        "declined_over5pp": 17
      },
      "unobserved_actual_sIoU": {
        "mean": -1.0242704434971093,
        "CI95": [
          -2.4730089722672597,
          0.13319285520949656
        ],
        "queries": 128,
        "parents": 128,
        "improved": 52,
        "harmed": 74,
        "unchanged": 2,
        "declined_over5pp": 16
      },
      "boundary_MAE_seconds": {
        "mean": 0.23540960244258885,
        "CI95": [
          0.11879435746794087,
          0.36934686900993724
        ],
        "queries": 128,
        "parents": 128,
        "improved": 15,
        "harmed": 32,
        "unchanged": 81,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.7617687344297792,
        "CI95": [
          -1.7773894426298589,
          0.20107409107319937
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 62,
        "unchanged": 20,
        "declined_over5pp": 12
      },
      "t": {
        "mean": -1.1391401321038128,
        "CI95": [
          -2.7706770845138236,
          0.4276057901899714
        ],
        "queries": 128,
        "parents": 128,
        "improved": 19,
        "harmed": 34,
        "unchanged": 75,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": -0.06846303745627587,
        "CI95": [
          -1.6139123269841773,
          1.4838018116257154
        ],
        "queries": 126,
        "parents": 126,
        "improved": 57,
        "harmed": 56,
        "unchanged": 13,
        "declined_over5pp": 10
      },
      "unobserved_actual_sIoU": {
        "mean": -0.3400765361428284,
        "CI95": [
          -0.6589222845680067,
          -0.043311669483223646
        ],
        "queries": 128,
        "parents": 128,
        "improved": 53,
        "harmed": 71,
        "unchanged": 4,
        "declined_over5pp": 3
      },
      "boundary_MAE_seconds": {
        "mean": 0.09831291822689256,
        "CI95": [
          -0.011017978143247498,
          0.21628188953121988
        ],
        "queries": 128,
        "parents": 128,
        "improved": 19,
        "harmed": 39,
        "unchanged": 70,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 26,
      "severe_v": 18,
      "severe_union_once": 35,
      "rescue_union_once": 75
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 19,
      "severe_v": 18,
      "severe_union_once": 25,
      "rescue_union_once": 11
    },
    "Source-CleanSource": {
      "severe_t": 20,
      "severe_v": 12,
      "severe_union_once": 20,
      "rescue_union_once": 13
    }
  }
}
