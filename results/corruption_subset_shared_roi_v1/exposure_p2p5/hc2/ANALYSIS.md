# hc2 / exposure_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.24445373243050014,
      "t": 0.4584528892515646,
      "s": 0.5258125521077841,
      "fixed_actual_sIoU": 0.5291601376642048,
      "observed_actual_sIoU": 0.5291926830128586,
      "unobserved_actual_sIoU": 0.5305422473394013,
      "boundary_MAE": 84.91015625,
      "start_error_seconds": 3.244223969799089,
      "end_error_seconds": 3.184417246924839,
      "interval_length_seconds": 6.081739495655202,
      "length_error_seconds": 3.417252842130461,
      "vIoU_at_03": 0.40625,
      "vIoU_at_05": 0.140625,
      "vIoU_ge_03": 0.40625,
      "vIoU_ge_05": 0.140625,
      "tIoU_ge_05": 0.4609375,
      "tIoU_eq_0": 0.140625,
      "boundary_MAE_seconds": 3.214320608361964
    },
    "FullMAP": {
      "v": 0.31479258468088533,
      "t": 0.5447876556649982,
      "s": 0.5512703526648485,
      "fixed_actual_sIoU": 0.553871306530624,
      "observed_actual_sIoU": 0.5866484512005467,
      "unobserved_actual_sIoU": 0.5482699192650387,
      "boundary_MAE": 78.46875,
      "start_error_seconds": 2.929392745466176,
      "end_error_seconds": 3.0081200797237093,
      "interval_length_seconds": 8.739634911911516,
      "length_error_seconds": 3.243377616453807,
      "vIoU_at_03": 0.4921875,
      "vIoU_at_05": 0.234375,
      "vIoU_ge_03": 0.4921875,
      "vIoU_ge_05": 0.234375,
      "tIoU_ge_05": 0.578125,
      "tIoU_eq_0": 0.09375,
      "boundary_MAE_seconds": 2.968756412594943
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
        "mean": 7.033885225038522,
        "CI95": [
          4.507968455191613,
          9.618156790791309
        ],
        "queries": 128,
        "parents": 128,
        "improved": 85,
        "harmed": 30,
        "unchanged": 13,
        "declined_over5pp": 19
      },
      "t": {
        "mean": 8.633476641343366,
        "CI95": [
          5.510886661788034,
          11.877503803002863
        ],
        "queries": 128,
        "parents": 128,
        "improved": 80,
        "harmed": 32,
        "unchanged": 16,
        "declined_over5pp": 21
      },
      "observed_actual_sIoU": {
        "mean": 5.745576818768815,
        "CI95": [
          1.5668062074193465,
          9.8852501383852
        ],
        "queries": 126,
        "parents": 126,
        "improved": 85,
        "harmed": 38,
        "unchanged": 3,
        "declined_over5pp": 30
      },
      "unobserved_actual_sIoU": {
        "mean": 1.7727671925637272,
        "CI95": [
          -0.8872083890406496,
          4.466988874390394
        ],
        "queries": 128,
        "parents": 128,
        "improved": 77,
        "harmed": 49,
        "unchanged": 2,
        "declined_over5pp": 26
      },
      "boundary_MAE_seconds": {
        "mean": -0.24556419576702132,
        "CI95": [
          -0.4314213136991853,
          -0.06481381184309142
        ],
        "queries": 128,
        "parents": 128,
        "improved": 78,
        "harmed": 42,
        "unchanged": 8,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.3120398734810363,
        "CI95": [
          -1.198517022734768,
          0.5751994904498526
        ],
        "queries": 128,
        "parents": 128,
        "improved": 50,
        "harmed": 64,
        "unchanged": 14,
        "declined_over5pp": 11
      },
      "t": {
        "mean": -0.4897827491148622,
        "CI95": [
          -1.4652987712664676,
          0.36765039519581966
        ],
        "queries": 128,
        "parents": 128,
        "improved": 19,
        "harmed": 22,
        "unchanged": 87,
        "declined_over5pp": 11
      },
      "observed_actual_sIoU": {
        "mean": 0.6702863513820444,
        "CI95": [
          -0.5580657075546998,
          2.003607365716466
        ],
        "queries": 126,
        "parents": 126,
        "improved": 51,
        "harmed": 55,
        "unchanged": 20,
        "declined_over5pp": 11
      },
      "unobserved_actual_sIoU": {
        "mean": 0.18998822284699154,
        "CI95": [
          -0.5897627213888414,
          1.0930194353230394
        ],
        "queries": 128,
        "parents": 128,
        "improved": 56,
        "harmed": 70,
        "unchanged": 2,
        "declined_over5pp": 9
      },
      "boundary_MAE_seconds": {
        "mean": 0.061813480399330295,
        "CI95": [
          -0.030502007855778644,
          0.16374148837308744
        ],
        "queries": 128,
        "parents": 128,
        "improved": 20,
        "harmed": 25,
        "unchanged": 83,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.6788796210703737,
        "CI95": [
          -1.3376026526417037,
          -0.11869610295502303
        ],
        "queries": 128,
        "parents": 128,
        "improved": 49,
        "harmed": 58,
        "unchanged": 21,
        "declined_over5pp": 10
      },
      "t": {
        "mean": -0.6641887542749702,
        "CI95": [
          -1.9005639685432922,
          0.45916173470883487
        ],
        "queries": 128,
        "parents": 128,
        "improved": 19,
        "harmed": 27,
        "unchanged": 82,
        "declined_over5pp": 13
      },
      "observed_actual_sIoU": {
        "mean": 0.31454492626085123,
        "CI95": [
          -0.5644180449245167,
          1.2493706903658166
        ],
        "queries": 126,
        "parents": 126,
        "improved": 61,
        "harmed": 50,
        "unchanged": 15,
        "declined_over5pp": 8
      },
      "unobserved_actual_sIoU": {
        "mean": -0.013099783513257753,
        "CI95": [
          -0.1802652435282707,
          0.14807015224414927
        ],
        "queries": 128,
        "parents": 128,
        "improved": 57,
        "harmed": 66,
        "unchanged": 5,
        "declined_over5pp": 0
      },
      "boundary_MAE_seconds": {
        "mean": 0.0529859029472067,
        "CI95": [
          -0.01974658332800971,
          0.12778038052404914
        ],
        "queries": 128,
        "parents": 128,
        "improved": 22,
        "harmed": 32,
        "unchanged": 74,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 21,
      "severe_v": 19,
      "severe_union_once": 32,
      "rescue_union_once": 78
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 11,
      "severe_v": 11,
      "severe_union_once": 13,
      "rescue_union_once": 12
    },
    "Source-CleanSource": {
      "severe_t": 13,
      "severe_v": 10,
      "severe_union_once": 13,
      "rescue_union_once": 9
    }
  }
}
