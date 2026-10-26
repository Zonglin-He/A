# vidstg / frame_freeze_p2p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.1327206449679839,
      "t": 0.3211870683186109,
      "s": 0.35942847675807243,
      "fixed_actual_sIoU": 0.36362091249961087,
      "observed_actual_sIoU": 0.37185357047943945,
      "unobserved_actual_sIoU": 0.36346950287238744,
      "boundary_MAE": 216.7890625,
      "start_error_seconds": 6.92835947937505,
      "end_error_seconds": 8.368693113994988,
      "interval_length_seconds": 7.813235480021541,
      "length_error_seconds": 7.570091615357773,
      "vIoU_at_03": 0.1640625,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.1640625,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.296875,
      "tIoU_eq_0": 0.234375,
      "boundary_MAE_seconds": 7.648526296685019
    },
    "FullMAP": {
      "v": 0.15091531271281478,
      "t": 0.34780443159021507,
      "s": 0.3870124814020493,
      "fixed_actual_sIoU": 0.3920926201741687,
      "observed_actual_sIoU": 0.4544606011823769,
      "unobserved_actual_sIoU": 0.38590104309468864,
      "boundary_MAE": 207.01953125,
      "start_error_seconds": 7.153186647948294,
      "end_error_seconds": 7.494062708524413,
      "interval_length_seconds": 11.488875051520873,
      "length_error_seconds": 8.193006588683854,
      "vIoU_at_03": 0.1875,
      "vIoU_at_05": 0.09375,
      "vIoU_ge_03": 0.1875,
      "vIoU_ge_05": 0.09375,
      "tIoU_ge_05": 0.3359375,
      "tIoU_eq_0": 0.171875,
      "boundary_MAE_seconds": 7.323624678236354
    },
    "CleanSource": {
      "v": 0.13322902725343239,
      "t": 0.32121534755059167,
      "s": 0.3605295478212531,
      "fixed_actual_sIoU": 0.3646913925512969,
      "observed_actual_sIoU": 0.3739578514313604,
      "unobserved_actual_sIoU": 0.3641293618234618,
      "boundary_MAE": 218.015625,
      "start_error_seconds": 7.0140726018788735,
      "end_error_seconds": 8.36576053156482,
      "interval_length_seconds": 7.829036650087549,
      "length_error_seconds": 7.678825410639762,
      "vIoU_at_03": 0.1640625,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.1640625,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.296875,
      "tIoU_eq_0": 0.2265625,
      "boundary_MAE_seconds": 7.689916566721846
    },
    "CleanFullMAP": {
      "v": 0.15339259040681869,
      "t": 0.3520243473184055,
      "s": 0.3839284008532412,
      "fixed_actual_sIoU": 0.3881660693441731,
      "observed_actual_sIoU": 0.45197783821206805,
      "unobserved_actual_sIoU": 0.38207574656684784,
      "boundary_MAE": 205.5625,
      "start_error_seconds": 7.117694327111833,
      "end_error_seconds": 7.42500476581608,
      "interval_length_seconds": 11.430284426520874,
      "length_error_seconds": 8.085201116805726,
      "vIoU_at_03": 0.1953125,
      "vIoU_at_05": 0.1015625,
      "vIoU_ge_03": 0.1953125,
      "vIoU_ge_05": 0.1015625,
      "tIoU_ge_05": 0.3359375,
      "tIoU_eq_0": 0.171875,
      "boundary_MAE_seconds": 7.271349546463956
    }
  },
  "paired": {
    "FullMAP-Source": {
      "v": {
        "mean": 1.819466774483089,
        "CI95": [
          -0.295437144549721,
          3.9371166360422447
        ],
        "queries": 128,
        "parents": 128,
        "improved": 65,
        "harmed": 37,
        "unchanged": 26,
        "declined_over5pp": 22
      },
      "t": {
        "mean": 2.6617363271604164,
        "CI95": [
          0.3950861812923599,
          5.1962247741080665
        ],
        "queries": 128,
        "parents": 128,
        "improved": 57,
        "harmed": 44,
        "unchanged": 27,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 8.260703070293745,
        "CI95": [
          2.925565725276416,
          13.683707815549326
        ],
        "queries": 104,
        "parents": 104,
        "improved": 70,
        "harmed": 31,
        "unchanged": 3,
        "declined_over5pp": 27
      },
      "unobserved_actual_sIoU": {
        "mean": 2.2431540222301285,
        "CI95": [
          -1.9256513742947652,
          6.387704370797848
        ],
        "queries": 128,
        "parents": 128,
        "improved": 75,
        "harmed": 50,
        "unchanged": 3,
        "declined_over5pp": 32
      },
      "boundary_MAE_seconds": {
        "mean": -0.3249016184486657,
        "CI95": [
          -0.9446828685143358,
          0.2259001501608022
        ],
        "queries": 128,
        "parents": 128,
        "improved": 60,
        "harmed": 58,
        "unchanged": 10,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.24772776940039165,
        "CI95": [
          -0.740495552270352,
          0.08186380589741937
        ],
        "queries": 128,
        "parents": 128,
        "improved": 37,
        "harmed": 31,
        "unchanged": 60,
        "declined_over5pp": 3
      },
      "t": {
        "mean": -0.4219915728190428,
        "CI95": [
          -0.8734097196718977,
          -0.06859881255868476
        ],
        "queries": 128,
        "parents": 128,
        "improved": 11,
        "harmed": 14,
        "unchanged": 103,
        "declined_over5pp": 5
      },
      "observed_actual_sIoU": {
        "mean": -0.08670664246838072,
        "CI95": [
          -0.5323238954950008,
          0.3982617281002115
        ],
        "queries": 103,
        "parents": 103,
        "improved": 23,
        "harmed": 35,
        "unchanged": 45,
        "declined_over5pp": 2
      },
      "unobserved_actual_sIoU": {
        "mean": 0.38252965278408035,
        "CI95": [
          -0.3092195464607748,
          1.2737147853304256
        ],
        "queries": 128,
        "parents": 128,
        "improved": 49,
        "harmed": 34,
        "unchanged": 45,
        "declined_over5pp": 3
      },
      "boundary_MAE_seconds": {
        "mean": 0.05227513177239746,
        "CI95": [
          -0.014934698854127435,
          0.1516304475276057
        ],
        "queries": 128,
        "parents": 128,
        "improved": 14,
        "harmed": 17,
        "unchanged": 97,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.05083822854485274,
        "CI95": [
          -0.2522052278424061,
          0.13034938281860156
        ],
        "queries": 128,
        "parents": 128,
        "improved": 30,
        "harmed": 34,
        "unchanged": 64,
        "declined_over5pp": 1
      },
      "t": {
        "mean": -0.002827923198075663,
        "CI95": [
          -0.28399362486288343,
          0.2934041553439836
        ],
        "queries": 128,
        "parents": 128,
        "improved": 8,
        "harmed": 7,
        "unchanged": 113,
        "declined_over5pp": 2
      },
      "observed_actual_sIoU": {
        "mean": -0.15092226669401151,
        "CI95": [
          -0.3357506322990914,
          -0.018321450324571564
        ],
        "queries": 103,
        "parents": 103,
        "improved": 28,
        "harmed": 36,
        "unchanged": 39,
        "declined_over5pp": 1
      },
      "unobserved_actual_sIoU": {
        "mean": -0.06598589510743566,
        "CI95": [
          -0.22989968459752738,
          0.07660328498671373
        ],
        "queries": 128,
        "parents": 128,
        "improved": 44,
        "harmed": 47,
        "unchanged": 37,
        "declined_over5pp": 1
      },
      "boundary_MAE_seconds": {
        "mean": -0.041390270036826944,
        "CI95": [
          -0.1244639918679036,
          0.021360598123994583
        ],
        "queries": 128,
        "parents": 128,
        "improved": 13,
        "harmed": 11,
        "unchanged": 104,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 23,
      "severe_v": 22,
      "severe_union_once": 36,
      "rescue_union_once": 57
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 5,
      "severe_v": 3,
      "severe_union_once": 5,
      "rescue_union_once": 0
    },
    "Source-CleanSource": {
      "severe_t": 2,
      "severe_v": 1,
      "severe_union_once": 2,
      "rescue_union_once": 1
    }
  }
}
