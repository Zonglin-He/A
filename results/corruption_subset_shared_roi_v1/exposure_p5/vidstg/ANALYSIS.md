# vidstg / exposure_p5

Fixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.

Full minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.

{
  "means": {
    "Source": {
      "v": 0.13246383117896987,
      "t": 0.32505940216528706,
      "s": 0.36010603803280195,
      "fixed_actual_sIoU": 0.3642037514569978,
      "observed_actual_sIoU": 0.3687500726845167,
      "unobserved_actual_sIoU": 0.36355852596482063,
      "boundary_MAE": 216.69921875,
      "start_error_seconds": 7.033243670603849,
      "end_error_seconds": 8.22665208429341,
      "interval_length_seconds": 7.739717550757833,
      "length_error_seconds": 7.518118919621482,
      "vIoU_at_03": 0.15625,
      "vIoU_at_05": 0.0390625,
      "vIoU_ge_03": 0.15625,
      "vIoU_ge_05": 0.0390625,
      "tIoU_ge_05": 0.3046875,
      "tIoU_eq_0": 0.2265625,
      "boundary_MAE_seconds": 7.629947877448629
    },
    "FullMAP": {
      "v": 0.1532114076580896,
      "t": 0.34938117353989795,
      "s": 0.3873348560982828,
      "fixed_actual_sIoU": 0.39259649378714834,
      "observed_actual_sIoU": 0.45583871354562466,
      "unobserved_actual_sIoU": 0.3868919935963571,
      "boundary_MAE": 206.6875,
      "start_error_seconds": 7.267466352437525,
      "end_error_seconds": 7.364231202518241,
      "interval_length_seconds": 11.277992913715845,
      "length_error_seconds": 8.235606616339835,
      "vIoU_at_03": 0.1953125,
      "vIoU_at_05": 0.1015625,
      "vIoU_ge_03": 0.1953125,
      "vIoU_ge_05": 0.1015625,
      "tIoU_ge_05": 0.3359375,
      "tIoU_eq_0": 0.171875,
      "boundary_MAE_seconds": 7.315848777477883
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
        "mean": 2.0747576479119716,
        "CI95": [
          -0.06218038370820829,
          4.289896499647291
        ],
        "queries": 128,
        "parents": 128,
        "improved": 69,
        "harmed": 33,
        "unchanged": 26,
        "declined_over5pp": 20
      },
      "t": {
        "mean": 2.432177137461079,
        "CI95": [
          0.29939334623431685,
          4.864971524648767
        ],
        "queries": 128,
        "parents": 128,
        "improved": 55,
        "harmed": 46,
        "unchanged": 27,
        "declined_over5pp": 23
      },
      "observed_actual_sIoU": {
        "mean": 8.7088640861108,
        "CI95": [
          3.2972176787087726,
          14.168062814255489
        ],
        "queries": 102,
        "parents": 102,
        "improved": 67,
        "harmed": 32,
        "unchanged": 3,
        "declined_over5pp": 25
      },
      "unobserved_actual_sIoU": {
        "mean": 2.3333467631536484,
        "CI95": [
          -1.761642113319553,
          6.372134659168211
        ],
        "queries": 128,
        "parents": 128,
        "improved": 75,
        "harmed": 49,
        "unchanged": 4,
        "declined_over5pp": 32
      },
      "boundary_MAE_seconds": {
        "mean": -0.31409909997074725,
        "CI95": [
          -0.9603995867458972,
          0.2518375943605049
        ],
        "queries": 128,
        "parents": 128,
        "improved": 61,
        "harmed": 60,
        "unchanged": 7,
        "declined_over5pp": null
      }
    },
    "FullMAP-CleanFullMAP": {
      "v": {
        "mean": -0.018118274872908513,
        "CI95": [
          -0.4723625082796096,
          0.4497094832377448
        ],
        "queries": 128,
        "parents": 128,
        "improved": 34,
        "harmed": 35,
        "unchanged": 59,
        "declined_over5pp": 4
      },
      "t": {
        "mean": -0.26431737785076015,
        "CI95": [
          -1.0705724323755161,
          0.476111747735212
        ],
        "queries": 128,
        "parents": 128,
        "improved": 20,
        "harmed": 24,
        "unchanged": 84,
        "declined_over5pp": 10
      },
      "observed_actual_sIoU": {
        "mean": 0.846599849362206,
        "CI95": [
          -0.08327471615397247,
          1.9181636687663755
        ],
        "queries": 102,
        "parents": 102,
        "improved": 29,
        "harmed": 34,
        "unchanged": 39,
        "declined_over5pp": 5
      },
      "unobserved_actual_sIoU": {
        "mean": 0.4816247029509212,
        "CI95": [
          -0.2812614567625586,
          1.3986713661563899
        ],
        "queries": 128,
        "parents": 128,
        "improved": 40,
        "harmed": 43,
        "unchanged": 45,
        "declined_over5pp": 7
      },
      "boundary_MAE_seconds": {
        "mean": 0.04449923101392676,
        "CI95": [
          -0.10838964395176998,
          0.2163373513562721
        ],
        "queries": 128,
        "parents": 128,
        "improved": 26,
        "harmed": 31,
        "unchanged": 71,
        "declined_over5pp": null
      }
    },
    "Source-CleanSource": {
      "v": {
        "mean": -0.07651960744625233,
        "CI95": [
          -0.6161260627653597,
          0.36032295237311307
        ],
        "queries": 128,
        "parents": 128,
        "improved": 28,
        "harmed": 38,
        "unchanged": 62,
        "declined_over5pp": 2
      },
      "t": {
        "mean": 0.3844054614695445,
        "CI95": [
          -0.9100217263348586,
          1.8056112390977492
        ],
        "queries": 128,
        "parents": 128,
        "improved": 19,
        "harmed": 14,
        "unchanged": 95,
        "declined_over5pp": 7
      },
      "observed_actual_sIoU": {
        "mean": -0.17837598077702643,
        "CI95": [
          -0.5733030908317569,
          0.17636260933902567
        ],
        "queries": 102,
        "parents": 102,
        "improved": 29,
        "harmed": 36,
        "unchanged": 37,
        "declined_over5pp": 2
      },
      "unobserved_actual_sIoU": {
        "mean": -0.057083585864116236,
        "CI95": [
          -0.25121494422754376,
          0.11752667182161393
        ],
        "queries": 128,
        "parents": 128,
        "improved": 44,
        "harmed": 48,
        "unchanged": 36,
        "declined_over5pp": 1
      },
      "boundary_MAE_seconds": {
        "mean": -0.0599686892732161,
        "CI95": [
          -0.23971063469961623,
          0.10808699913301331
        ],
        "queries": 128,
        "parents": 128,
        "improved": 24,
        "harmed": 20,
        "unchanged": 84,
        "declined_over5pp": null
      }
    }
  },
  "tails": {
    "FullMAP-Source": {
      "severe_t": 23,
      "severe_v": 20,
      "severe_union_once": 34,
      "rescue_union_once": 55
    },
    "FullMAP-CleanFullMAP": {
      "severe_t": 10,
      "severe_v": 4,
      "severe_union_once": 12,
      "rescue_union_once": 10
    },
    "Source-CleanSource": {
      "severe_t": 7,
      "severe_v": 2,
      "severe_union_once": 7,
      "rescue_union_once": 10
    }
  }
}
