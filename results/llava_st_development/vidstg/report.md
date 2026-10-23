# LLaVA-ST backbone: complete target population

Same merged checkpoint on both domains, no strict source-only cross-domain claim. Source/Ours share native generated time, no Global6/MBR. Exposed128 development included in full. Invalid outputs retained, never replace query.

{
  "status": "complete",
  "dataset": "vidstg",
  "queries": 10303,
  "parameters": null,
  "means": {
    "Source": {
      "v": 12.43302008053614,
      "s": 27.42877829988915,
      "t": 42.67798448703735
    },
    "Ours": {
      "v": 15.363699975228279,
      "s": 29.882846474568574,
      "t": 42.67798448703735
    }
  },
  "unobserved_sIoU": {
    "Source": 27.411011105567802,
    "Ours": 29.611171014754262
  },
  "observed_sIoU": {
    "Source": 28.468248608575102,
    "Ours": 42.69550636265343
  },
  "invalid_native_outputs_retained": 0,
  "metric_denominators": {
    "Source": {
      "main": 10303,
      "unobserved": 10303,
      "observed": 8593
    },
    "Ours": {
      "main": 10303,
      "unobserved": 10303,
      "observed": 8593
    }
  },
  "loss_over5pp_queries": 1397,
  "prediction_seal_sha256": "9c577357d0635761f2afcd5390bc4ab702e47952255a579dd8e125f757bd8cfb",
  "fixed_native_time": true,
  "no_independent_unexposed_test_claim": true,
  "paired": {
    "s": {
      "queries": 10303,
      "parents": 732,
      "mean_pp": 2.4540681746794206,
      "CI95_pp": [
        1.648729967450178,
        3.266021210293868
      ],
      "equal_parent_mean_pp": 3.5228153302103595,
      "wins": 5615,
      "losses": 4050,
      "ties": 638,
      "below_minus5": 2863
    },
    "v": {
      "queries": 10303,
      "parents": 732,
      "mean_pp": 2.930679894692136,
      "CI95_pp": [
        2.431024719008428,
        3.4441656076208513
      ],
      "equal_parent_mean_pp": 3.73378844733522,
      "wins": 4647,
      "losses": 2677,
      "ties": 2979,
      "below_minus5": 1397
    },
    "unobserved": {
      "queries": 10303,
      "parents": 732,
      "mean_pp": 2.200159909186456,
      "CI95_pp": [
        1.3986681657305757,
        3.0075422969254415
      ],
      "equal_parent_mean_pp": 3.239069974389263,
      "wins": 5525,
      "losses": 4109,
      "ties": 669,
      "below_minus5": 2902
    }
  }
}
