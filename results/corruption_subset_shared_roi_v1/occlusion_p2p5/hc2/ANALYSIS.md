# hc2 / occlusion_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.2419591666854492,
      "t": 0.4501733386043668,
      "s": 0.522885572975377,
      "fixed_actual_sIoU": 0.5261330019691532,
      "observed_actual_sIoU": 0.5169628062276213,
      "unobserved_actual_sIoU": 0.5282199905072116,
      "boundary_MAE": 86.73046875,
      "start_error_seconds": 3.334024169205251,
      "end_error_seconds": 3.233796683504468,
      "interval_length_seconds": 6.040221506664458,
      "length_error_seconds": 3.3872694596597537,
      "vIoU_at_03": 0.40625,
      "vIoU_at_05": 0.1328125,
      "vIoU_ge_03": 0.40625,
      "vIoU_ge_05": 0.1328125,
      "tIoU_ge_05": 0.4453125,
      "tIoU_eq_0": 0.1484375,
      "boundary_MAE_seconds": 3.2839104263548595
    },
    "FullMAP": {
      "v": 0.3149851031150901,
      "t": 0.5395266568343753,
      "s": 0.5483095870033292,
      "fixed_actual_sIoU": 0.5493928156702881,
      "observed_actual_sIoU": 0.5742579586484355,
      "unobserved_actual_sIoU": 0.544507165831339,
      "boundary_MAE": 79.578125,
      "start_error_seconds": 2.8903884116836576,
      "end_error_seconds": 3.1156008555238373,
      "interval_length_seconds": 8.82833946883077,
      "length_error_seconds": 3.381266984828723,
      "vIoU_at_03": 0.5234375,
      "vIoU_at_05": 0.2421875,
      "vIoU_ge_03": 0.5234375,
      "vIoU_ge_05": 0.2421875,
      "tIoU_ge_05": 0.546875,
      "tIoU_eq_0": 0.09375,
      "boundary_MAE_seconds": 3.002994633603748
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
        "mean": 7.302593642964092,
        "CI95": [
          4.719930821192193,
          9.846306873770928
        ],
        "queries": 128,
        "parents": 128,
        "improved": 87,
        "harmed": 27,
        "unchanged": 14,
        "declined_over5pp": 16
      },
      "t": {
        "mean": 8.935331823000844,
        "CI95": [
          5.83713467672527,
          12.069637818587172
        ],
        "queries": 128,
        "parents": 128,
        "improved": 83,
        "harmed": 30,
        "unchanged": 15,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": 5.729515242081431,
        "CI95": [
          1.616509492774972,
          9.851751501960406
        ],
        "queries": 126,
        "parents": 126,
        "improved": 82,
        "harmed": 38,
        "unchanged": 6,
        "declined_over5pp": 27
      },
      "unobserved_actual_sIoU": {
        "mean": 1.6287175324127237,
        "CI95": [
          -1.0895149083327664,
          4.344291112420668
        ],
        "queries": 128,
        "parents": 128,
        "improved": 75,
        "harmed": 51,
        "unchanged": 2,
        "declined_over5pp": 28
      },
      "boundary_MAE_seconds": {
        "mean": -0.280915792751112,
        "CI95": [
          -0.472459362762721,
          -0.09635680348192362
        ],
        "queries": 128,
        "parents": 128,
        "improved": 78,
        "harmed": 46,
        "unchanged": 4,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.2927880300605614,
        "CI95": [
          -1.2317260265639531,
          0.6780228810166178
        ],
        "queries": 128,
        "parents": 128,
        "improved": 58,
        "harmed": 56,
        "unchanged": 14,
        "declined_over5pp": 16
      },
      "t": {
        "mean": -1.015882632177166,
        "CI95": [
          -2.211025317006854,
          0.0037334076701029486
        ],
        "queries": 128,
        "parents": 128,
        "improved": 16,
        "harmed": 20,
        "unchanged": 92,
        "declined_over5pp": 13
      },
      "observed_actual_sIoU": {
        "mean": -0.5687629038290843,
        "CI95": [
          -2.1845154167324057,
          0.9981693217164074
        ],
        "queries": 126,
        "parents": 126,
        "improved": 52,
        "harmed": 55,
        "unchanged": 19,
        "declined_over5pp": 14
      },
      "unobserved_actual_sIoU": {
        "mean": -0.1862871205229765,
        "CI95": [
          -1.2816224788336723,
          0.9607876200587511
        ],
        "queries": 128,
        "parents": 128,
        "improved": 62,
        "harmed": 64,
        "unchanged": 2,
        "declined_over5pp": 14
      },
      "boundary_MAE_seconds": {
        "mean": 0.0960517014081353,
        "CI95": [
          0.004296648802680651,
          0.19837093769021186
        ],
        "queries": 128,
        "parents": 128,
        "improved": 18,
        "harmed": 21,
        "unchanged": 89,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.928336195575468,
        "CI95": [
          -1.539957676988447,
          -0.39674279315202515
        ],
        "queries": 128,
        "parents": 128,
        "improved": 47,
        "harmed": 60,
        "unchanged": 21,
        "declined_over5pp": 12
      },
      "t": {
        "mean": -1.492143818994752,
        "CI95": [
          -2.562235660883631,
          -0.5938195584093396
        ],
        "queries": 128,
        "parents": 128,
        "improved": 11,
        "harmed": 27,
        "unchanged": 90,
        "declined_over5pp": 17
      },
      "observed_actual_sIoU": {
        "mean": -0.9084427522628921,
        "CI95": [
          -2.056362046706337,
          0.01617011676533182
        ],
        "queries": 126,
        "parents": 126,
        "improved": 44,
        "harmed": 65,
        "unchanged": 17,
        "declined_over5pp": 8
      },
      "unobserved_actual_sIoU": {
        "mean": -0.24532546673222277,
        "CI95": [
          -0.4952913963350931,
          -0.0064835811158634425
        ],
        "queries": 128,
        "parents": 128,
        "improved": 59,
        "harmed": 63,
        "unchanged": 6,
        "declined_over5pp": 3
      },
      "boundary_MAE_seconds": {
        "mean": 0.12257572094010238,
        "CI95": [
          0.052242154357968584,
          0.21455027033975782
        ],
        "queries": 128,
        "parents": 128,
        "improved": 12,
        "harmed": 30,
        "unchanged": 86,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 20,
      "severe_v": 16,
      "severe_union_once": 29,
      "rescue_union_once": 80
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 13,
      "severe_v": 16,
      "severe_union_once": 21,
      "rescue_union_once": 10
    },
    "Source-CleanSource": {
      "severe_t": 17,
      "severe_v": 12,
      "severe_union_once": 17,
      "rescue_union_once": 3
    }
  }
}
