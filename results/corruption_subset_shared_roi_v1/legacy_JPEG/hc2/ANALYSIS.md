# hc2 / legacy_JPEG

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.24940262965753124,
      "t": 0.4620245235535264,
      "s": 0.5250682700229862,
      "fixed_actual_sIoU": 0.5300401360069578,
      "observed_actual_sIoU": 0.5315313256403883,
      "unobserved_actual_sIoU": 0.5306936013867054,
      "boundary_MAE": 83.8125,
      "start_error_seconds": 3.1818202713021275,
      "end_error_seconds": 3.1535589095541203,
      "interval_length_seconds": 6.488731223044344,
      "length_error_seconds": 3.516141066479649,
      "vIoU_at_03": 0.4296875,
      "vIoU_at_05": 0.1328125,
      "vIoU_ge_03": 0.4296875,
      "vIoU_ge_05": 0.1328125,
      "tIoU_ge_05": 0.4609375,
      "tIoU_eq_0": 0.1171875,
      "boundary_MAE_seconds": 3.1676895904281235
    },
    "FullMAP": {
      "v": 0.31318359625793135,
      "t": 0.5450799252898586,
      "s": 0.5447032841345666,
      "fixed_actual_sIoU": 0.5468242304700857,
      "observed_actual_sIoU": 0.577967600762943,
      "unobserved_actual_sIoU": 0.5419880573662066,
      "boundary_MAE": 77.57421875,
      "start_error_seconds": 2.825340943742483,
      "end_error_seconds": 3.001495373393376,
      "interval_length_seconds": 8.993251919753272,
      "length_error_seconds": 3.1777989379071547,
      "vIoU_at_03": 0.4921875,
      "vIoU_at_05": 0.2421875,
      "vIoU_ge_03": 0.4921875,
      "vIoU_ge_05": 0.2421875,
      "tIoU_ge_05": 0.578125,
      "tIoU_eq_0": 0.0859375,
      "boundary_MAE_seconds": 2.9134181585679295
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
        "mean": 6.378096660040015,
        "CI95": [
          3.7672102540090355,
          8.88875818321871
        ],
        "queries": 128,
        "parents": 128,
        "improved": 87,
        "harmed": 31,
        "unchanged": 10,
        "declined_over5pp": 17
      },
      "t": {
        "mean": 8.30554017363321,
        "CI95": [
          5.015108356123424,
          11.515037398113952
        ],
        "queries": 128,
        "parents": 128,
        "improved": 83,
        "harmed": 30,
        "unchanged": 15,
        "declined_over5pp": 17
      },
      "observed_actual_sIoU": {
        "mean": 4.643627512255462,
        "CI95": [
          0.8103535679532143,
          8.379447420298051
        ],
        "queries": 126,
        "parents": 126,
        "improved": 83,
        "harmed": 37,
        "unchanged": 6,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 1.1294455979501248,
        "CI95": [
          -1.6677461060823855,
          3.8255088146321476
        ],
        "queries": 128,
        "parents": 128,
        "improved": 84,
        "harmed": 44,
        "unchanged": 0,
        "declined_over5pp": 25
      },
      "boundary_MAE_seconds": {
        "mean": -0.2542714318601945,
        "CI95": [
          -0.4512181183137597,
          -0.049704932258853335
        ],
        "queries": 128,
        "parents": 128,
        "improved": 70,
        "harmed": 49,
        "unchanged": 9,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.4729387157764348,
        "CI95": [
          -2.303473037681998,
          1.2254499455880183
        ],
        "queries": 128,
        "parents": 128,
        "improved": 56,
        "harmed": 61,
        "unchanged": 11,
        "declined_over5pp": 22
      },
      "t": {
        "mean": -0.4605557866288313,
        "CI95": [
          -2.423073345722423,
          1.3495312374184671
        ],
        "queries": 128,
        "parents": 128,
        "improved": 36,
        "harmed": 36,
        "unchanged": 56,
        "declined_over5pp": 20
      },
      "observed_actual_sIoU": {
        "mean": -0.19779869237834422,
        "CI95": [
          -2.7536939299500993,
          2.3408719049399593
        ],
        "queries": 126,
        "parents": 126,
        "improved": 70,
        "harmed": 51,
        "unchanged": 5,
        "declined_over5pp": 21
      },
      "unobserved_actual_sIoU": {
        "mean": -0.4381979670361978,
        "CI95": [
          -2.282174850600071,
          1.3003193964031405
        ],
        "queries": 128,
        "parents": 128,
        "improved": 61,
        "harmed": 67,
        "unchanged": 0,
        "declined_over5pp": 23
      },
      "boundary_MAE_seconds": {
        "mean": 0.0064752263723169565,
        "CI95": [
          -0.16714023558450614,
          0.18461243343727107
        ],
        "queries": 128,
        "parents": 128,
        "improved": 38,
        "harmed": 36,
        "unchanged": 54,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.18398989836726437,
        "CI95": [
          -1.6243742036070037,
          1.0678960279501775
        ],
        "queries": 128,
        "parents": 128,
        "improved": 57,
        "harmed": 55,
        "unchanged": 16,
        "declined_over5pp": 7
      },
      "t": {
        "mean": -0.3070253240787834,
        "CI95": [
          -2.206023123691765,
          1.3970285938066227
        ],
        "queries": 128,
        "parents": 128,
        "improved": 40,
        "harmed": 35,
        "unchanged": 53,
        "declined_over5pp": 18
      },
      "observed_actual_sIoU": {
        "mean": 0.5484091890138154,
        "CI95": [
          -1.1873073992791268,
          2.3308814883283384
        ],
        "queries": 126,
        "parents": 126,
        "improved": 62,
        "harmed": 56,
        "unchanged": 8,
        "declined_over5pp": 11
      },
      "unobserved_actual_sIoU": {
        "mean": 0.0020356212171544485,
        "CI95": [
          -0.5535418341583557,
          0.603814970712547
        ],
        "queries": 128,
        "parents": 128,
        "improved": 61,
        "harmed": 65,
        "unchanged": 2,
        "declined_over5pp": 3
      },
      "boundary_MAE_seconds": {
        "mean": 0.006354885013366597,
        "CI95": [
          -0.143989971377027,
          0.15987061516076131
        ],
        "queries": 128,
        "parents": 128,
        "improved": 44,
        "harmed": 37,
        "unchanged": 47,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 17,
      "severe_v": 17,
      "severe_union_once": 28,
      "rescue_union_once": 71
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 20,
      "severe_v": 22,
      "severe_union_once": 30,
      "rescue_union_once": 26
    },
    "Source-CleanSource": {
      "severe_t": 18,
      "severe_v": 7,
      "severe_union_once": 18,
      "rescue_union_once": 29
    }
  }
}
