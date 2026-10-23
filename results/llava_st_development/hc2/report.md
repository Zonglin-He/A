# LLaVA-ST backbone: complete target population

Same merged checkpoint on both domains, no strict source-only cross-domain claim. Source/Ours share native generated time, no Global6/MBR. Exposed128 development included in full. Invalid outputs retained, never replace query.

{
  "status": "complete",
  "dataset": "hc2",
  "queries": 1902,
  "parameters": null,
  "means": {
    "Source": {
      "v": 9.70473746155963,
      "s": 38.41536476934518,
      "t": 21.73059264258822
    },
    "Ours": {
      "v": 11.794342928805106,
      "s": 42.08435389190757,
      "t": 21.73059264258822
    }
  },
  "unobserved_sIoU": {
    "Source": 38.339155587045006,
    "Ours": 41.84324549548988
  },
  "observed_sIoU": {
    "Source": 41.69093146198025,
    "Ours": 52.090488998276776
  },
  "invalid_native_outputs_retained": 0,
  "metric_denominators": {
    "Source": {
      "main": 1902,
      "unobserved": 1902,
      "observed": 1857
    },
    "Ours": {
      "main": 1902,
      "unobserved": 1902,
      "observed": 1857
    }
  },
  "loss_over5pp_queries": 134,
  "prediction_seal_sha256": "670636e0e1ec0b011a9bc805d45f29cdbb662d238076b445cac1f26ebcc58af8",
  "fixed_native_time": true,
  "no_independent_unexposed_test_claim": true,
  "paired": {
    "s": {
      "queries": 1902,
      "parents": 227,
      "mean_pp": 3.6689891225623805,
      "CI95_pp": [
        2.9806218064478616,
        4.348817394343746
      ],
      "equal_parent_mean_pp": 3.6893021225147935,
      "wins": 1332,
      "losses": 553,
      "ties": 17,
      "below_minus5": 304
    },
    "v": {
      "queries": 1902,
      "parents": 227,
      "mean_pp": 2.0896054672454754,
      "CI95_pp": [
        1.7368331757256519,
        2.4369073134123194
      ],
      "equal_parent_mean_pp": 2.2702043608606988,
      "wins": 1071,
      "losses": 418,
      "ties": 413,
      "below_minus5": 134
    },
    "unobserved": {
      "queries": 1902,
      "parents": 227,
      "mean_pp": 3.504089908444874,
      "CI95_pp": [
        2.8197608573763575,
        4.180088076995984
      ],
      "equal_parent_mean_pp": 3.517676950434842,
      "wins": 1326,
      "losses": 559,
      "ties": 17,
      "below_minus5": 308
    }
  }
}
