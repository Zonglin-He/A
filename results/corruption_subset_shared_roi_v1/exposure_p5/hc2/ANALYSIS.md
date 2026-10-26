# hc2 / exposure_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.24020776105502412,
      "t": 0.45113651737291266,
      "s": 0.5250497308721658,
      "fixed_actual_sIoU": 0.5288243553041108,
      "observed_actual_sIoU": 0.5274813224026589,
      "unobserved_actual_sIoU": 0.5308463331686373,
      "boundary_MAE": 86.0,
      "start_error_seconds": 3.351156669460683,
      "end_error_seconds": 3.164621401842956,
      "interval_length_seconds": 5.982163646275112,
      "length_error_seconds": 3.35936795748272,
      "vIoU_at_03": 0.40625,
      "vIoU_at_05": 0.1328125,
      "vIoU_ge_03": 0.40625,
      "vIoU_ge_05": 0.1328125,
      "tIoU_ge_05": 0.4609375,
      "tIoU_eq_0": 0.1484375,
      "boundary_MAE_seconds": 3.2578890356518198
    },
    "FullMAP": {
      "v": 0.3110598776522888,
      "t": 0.5406396180791909,
      "s": 0.5522395864043855,
      "fixed_actual_sIoU": 0.5543979352266026,
      "observed_actual_sIoU": 0.5847094785583357,
      "unobserved_actual_sIoU": 0.5496594300273396,
      "boundary_MAE": 78.83984375,
      "start_error_seconds": 2.938683728028921,
      "end_error_seconds": 3.026627690645792,
      "interval_length_seconds": 8.804231520223594,
      "length_error_seconds": 3.243227897951959,
      "vIoU_at_03": 0.5078125,
      "vIoU_at_05": 0.2109375,
      "vIoU_ge_03": 0.5078125,
      "vIoU_ge_05": 0.2109375,
      "tIoU_ge_05": 0.5703125,
      "tIoU_eq_0": 0.09375,
      "boundary_MAE_seconds": 2.9826557093373567
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
        "mean": 7.085211659726472,
        "CI95": [
          4.231056648029194,
          9.96290489040786
        ],
        "queries": 128,
        "parents": 128,
        "improved": 87,
        "harmed": 28,
        "unchanged": 13,
        "declined_over5pp": 17
      },
      "t": {
        "mean": 8.950310070627825,
        "CI95": [
          5.688742682947667,
          12.328089797638393
        ],
        "queries": 128,
        "parents": 128,
        "improved": 80,
        "harmed": 33,
        "unchanged": 15,
        "declined_over5pp": 19
      },
      "observed_actual_sIoU": {
        "mean": 5.7228156155676615,
        "CI95": [
          1.36760228622583,
          10.100872062236618
        ],
        "queries": 126,
        "parents": 126,
        "improved": 86,
        "harmed": 37,
        "unchanged": 3,
        "declined_over5pp": 29
      },
      "unobserved_actual_sIoU": {
        "mean": 1.8813096858702383,
        "CI95": [
          -0.9516985251843735,
          4.749713273268385
        ],
        "queries": 128,
        "parents": 128,
        "improved": 81,
        "harmed": 45,
        "unchanged": 2,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": -0.27523332631446323,
        "CI95": [
          -0.47419289334698234,
          -0.07906212427453205
        ],
        "queries": 128,
        "parents": 128,
        "improved": 74,
        "harmed": 48,
        "unchanged": 6,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.6853105763406866,
        "CI95": [
          -1.9137813432559738,
          0.3940395952555178
        ],
        "queries": 128,
        "parents": 128,
        "improved": 52,
        "harmed": 63,
        "unchanged": 13,
        "declined_over5pp": 16
      },
      "t": {
        "mean": -0.9045865076955937,
        "CI95": [
          -2.3143161828301144,
          0.24487261755400894
        ],
        "queries": 128,
        "parents": 128,
        "improved": 19,
        "harmed": 25,
        "unchanged": 84,
        "declined_over5pp": 15
      },
      "observed_actual_sIoU": {
        "mean": 0.4763890871609324,
        "CI95": [
          -0.5364270079800104,
          1.636595264484957
        ],
        "queries": 126,
        "parents": 126,
        "improved": 56,
        "harmed": 55,
        "unchanged": 15,
        "declined_over5pp": 12
      },
      "unobserved_actual_sIoU": {
        "mean": 0.32893929907710345,
        "CI95": [
          -0.5161051000041113,
          1.1976893036554173
        ],
        "queries": 128,
        "parents": 128,
        "improved": 62,
        "harmed": 64,
        "unchanged": 2,
        "declined_over5pp": 10
      },
      "boundary_MAE_seconds": {
        "mean": 0.07571277714174397,
        "CI95": [
          -0.021317211180021375,
          0.18676025656900372
        ],
        "queries": 128,
        "parents": 128,
        "improved": 21,
        "harmed": 27,
        "unchanged": 80,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -1.103476758617973,
        "CI95": [
          -2.049924121427955,
          -0.22895611508067942
        ],
        "queries": 128,
        "parents": 128,
        "improved": 46,
        "harmed": 61,
        "unchanged": 21,
        "declined_over5pp": 15
      },
      "t": {
        "mean": -1.3958259421401626,
        "CI95": [
          -3.0086559084305375,
          0.0798227020438219
        ],
        "queries": 128,
        "parents": 128,
        "improved": 20,
        "harmed": 36,
        "unchanged": 72,
        "declined_over5pp": 21
      },
      "observed_actual_sIoU": {
        "mean": 0.14340886524089286,
        "CI95": [
          -0.6668925829940462,
          1.022994297731941
        ],
        "queries": 126,
        "parents": 126,
        "improved": 60,
        "harmed": 58,
        "unchanged": 8,
        "declined_over5pp": 10
      },
      "unobserved_actual_sIoU": {
        "mean": 0.017308799410342836,
        "CI95": [
          -0.21537369044221083,
          0.22210141212438442
        ],
        "queries": 128,
        "parents": 128,
        "improved": 55,
        "harmed": 69,
        "unchanged": 4,
        "declined_over5pp": 1
      },
      "boundary_MAE_seconds": {
        "mean": 0.09655433023706231,
        "CI95": [
          0.005066208230876188,
          0.20525526737030259
        ],
        "queries": 128,
        "parents": 128,
        "improved": 21,
        "harmed": 42,
        "unchanged": 65,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 19,
      "severe_v": 17,
      "severe_union_once": 29,
      "rescue_union_once": 76
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 15,
      "severe_v": 16,
      "severe_union_once": 18,
      "rescue_union_once": 13
    },
    "Source-CleanSource": {
      "severe_t": 21,
      "severe_v": 15,
      "severe_union_once": 21,
      "rescue_union_once": 11
    }
  }
}
