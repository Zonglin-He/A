# hc2 / frame_freeze_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.24775130909513737,
      "t": 0.462863066730958,
      "s": 0.5250744395317816,
      "fixed_actual_sIoU": 0.5280704605649333,
      "observed_actual_sIoU": 0.5172443819589992,
      "unobserved_actual_sIoU": 0.5308117139669521,
      "boundary_MAE": 84.1015625,
      "start_error_seconds": 3.178450944339072,
      "end_error_seconds": 3.1822055637110065,
      "interval_length_seconds": 6.26784489937287,
      "length_error_seconds": 3.287727112716346,
      "vIoU_at_03": 0.40625,
      "vIoU_at_05": 0.140625,
      "vIoU_ge_03": 0.40625,
      "vIoU_ge_05": 0.140625,
      "tIoU_ge_05": 0.484375,
      "tIoU_eq_0": 0.1484375,
      "boundary_MAE_seconds": 3.1803282540250395
    },
    "FullMAP": {
      "v": 0.3198713813379765,
      "t": 0.5513629047246632,
      "s": 0.5491278688482119,
      "fixed_actual_sIoU": 0.5507075791876277,
      "observed_actual_sIoU": 0.5745520982552926,
      "unobserved_actual_sIoU": 0.5456211839872501,
      "boundary_MAE": 76.11328125,
      "start_error_seconds": 2.7655876865400275,
      "end_error_seconds": 2.9889609479429926,
      "interval_length_seconds": 8.975522742878855,
      "length_error_seconds": 3.289286243677176,
      "vIoU_at_03": 0.5,
      "vIoU_at_05": 0.234375,
      "vIoU_ge_03": 0.5,
      "vIoU_ge_05": 0.234375,
      "tIoU_ge_05": 0.5859375,
      "tIoU_eq_0": 0.09375,
      "boundary_MAE_seconds": 2.8772743172415103
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
        "mean": 7.21200722428391,
        "CI95": [
          4.562566057415288,
          9.930792143640863
        ],
        "queries": 128,
        "parents": 128,
        "improved": 82,
        "harmed": 33,
        "unchanged": 13,
        "declined_over5pp": 18
      },
      "t": {
        "mean": 8.849983799370515,
        "CI95": [
          5.742267570440055,
          12.069025074339894
        ],
        "queries": 128,
        "parents": 128,
        "improved": 78,
        "harmed": 34,
        "unchanged": 16,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": 5.730771629629333,
        "CI95": [
          1.633118718673008,
          9.823425476111176
        ],
        "queries": 126,
        "parents": 126,
        "improved": 84,
        "harmed": 37,
        "unchanged": 5,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 1.4809470020297906,
        "CI95": [
          -1.2633299051797864,
          4.248619524764482
        ],
        "queries": 128,
        "parents": 128,
        "improved": 74,
        "harmed": 52,
        "unchanged": 2,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": -0.3030539367835292,
        "CI95": [
          -0.5032004466703026,
          -0.11244430797349776
        ],
        "queries": 128,
        "parents": 128,
        "improved": 73,
        "harmed": 48,
        "unchanged": 7,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": 0.19583979222807577,
        "CI95": [
          -0.53657227864049,
          1.0182166491842302
        ],
        "queries": 128,
        "parents": 128,
        "improved": 49,
        "harmed": 65,
        "unchanged": 14,
        "declined_over5pp": 5
      },
      "t": {
        "mean": 0.16774215685162852,
        "CI95": [
          -0.528516732024703,
          0.8179585386228948
        ],
        "queries": 128,
        "parents": 128,
        "improved": 15,
        "harmed": 10,
        "unchanged": 103,
        "declined_over5pp": 7
      },
      "observed_actual_sIoU": {
        "mean": -0.5393489431433666,
        "CI95": [
          -1.7209932944225195,
          0.6192471320100607
        ],
        "queries": 126,
        "parents": 126,
        "improved": 42,
        "harmed": 61,
        "unchanged": 23,
        "declined_over5pp": 9
      },
      "unobserved_actual_sIoU": {
        "mean": -0.07488530493185815,
        "CI95": [
          -0.8740068930342442,
          0.8503767738174767
        ],
        "queries": 128,
        "parents": 128,
        "improved": 53,
        "harmed": 73,
        "unchanged": 2,
        "declined_over5pp": 8
      },
      "boundary_MAE_seconds": {
        "mean": -0.029668614954102192,
        "CI95": [
          -0.11402097056690294,
          0.04802349551944985
        ],
        "queries": 128,
        "parents": 128,
        "improved": 17,
        "harmed": 11,
        "unchanged": 100,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.3491219546066502,
        "CI95": [
          -0.8588527797595764,
          0.11909000280649666
        ],
        "queries": 128,
        "parents": 128,
        "improved": 47,
        "harmed": 60,
        "unchanged": 21,
        "declined_over5pp": 7
      },
      "t": {
        "mean": -0.22317100633562822,
        "CI95": [
          -1.2035160753883767,
          0.6848478615037134
        ],
        "queries": 128,
        "parents": 128,
        "improved": 14,
        "harmed": 20,
        "unchanged": 94,
        "declined_over5pp": 8
      },
      "observed_actual_sIoU": {
        "mean": -0.880285179125078,
        "CI95": [
          -2.0568766255770488,
          -0.08117004718764745
        ],
        "queries": 126,
        "parents": 126,
        "improved": 52,
        "harmed": 57,
        "unchanged": 17,
        "declined_over5pp": 7
      },
      "unobserved_actual_sIoU": {
        "mean": 0.013846879241828392,
        "CI95": [
          -0.11316712017636764,
          0.1520907612236264
        ],
        "queries": 128,
        "parents": 128,
        "improved": 55,
        "harmed": 66,
        "unchanged": 7,
        "declined_over5pp": 0
      },
      "boundary_MAE_seconds": {
        "mean": 0.018993548610282168,
        "CI95": [
          -0.041446176574172,
          0.08239349684459393
        ],
        "queries": 128,
        "parents": 128,
        "improved": 14,
        "harmed": 22,
        "unchanged": 92,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 20,
      "severe_v": 18,
      "severe_union_once": 32,
      "rescue_union_once": 78
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 7,
      "severe_v": 5,
      "severe_union_once": 9,
      "rescue_union_once": 9
    },
    "Source-CleanSource": {
      "severe_t": 8,
      "severe_v": 7,
      "severe_union_once": 8,
      "rescue_union_once": 8
    }
  }
}
