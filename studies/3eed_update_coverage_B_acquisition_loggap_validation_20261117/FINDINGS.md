# R132 findings

{
  "no_combination": true,
  "cohort": 1798,
  "source_training": false,
  "new_GT_geometry": false,
  "FINAL_access": false,
  "robust_method_adopted": false,
  "factors": {
    "B3": {
      "comparisons": {
        "raw": {
          "mIoU_percent": 12.655460909888562,
          "gains_pp": {
            "B3_zero": -0.03139169361156391,
            "B3_direct": -0.018276617373047088,
            "B3_wrong": 0.09721273844139589,
            "partial_F": -0.07257835095626852
          }
        },
        "design_weighted": {
          "mIoU_percent": 12.636028155188768,
          "gains_pp": {
            "B3_zero": -0.0019939190072282287,
            "B3_direct": 0.011193898739003671,
            "B3_wrong": 0.11637693499716129,
            "partial_F": -0.04009157836385846
          }
        }
      },
      "retain_as_working_factor": false
    },
    "LOG": {
      "comparisons": {
        "raw": {
          "mIoU_percent": 12.828305077237795,
          "gains_pp": {
            "LOG_zero": 0.13304239294504683,
            "LOG_direct": 0.12102013538187606,
            "LOG_wrong": 0.22265156140150033,
            "partial_F": 0.10026581639296417
          }
        },
        "design_weighted": {
          "mIoU_percent": 12.793106902906024,
          "gains_pp": {
            "LOG_zero": 0.1625884745291799,
            "LOG_direct": 0.14895439611924743,
            "LOG_wrong": 0.253790189238142,
            "partial_F": 0.11698716935339704
          }
        }
      },
      "retain_as_working_factor": true
    }
  }
}

See REPORT.md for complete positive/negative task and process results. No compound method or follow-up experiment was run.
