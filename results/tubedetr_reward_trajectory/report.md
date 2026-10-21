# TubeDETR 真实decoder全60步路径复现

{
  "status": "complete",
  "queries": 16,
  "endpoints_bitexact_with_old_ROI": true,
  "mechanistic_replay_not_new_method": true,
  "no_step_selection_or_GT_early_stopping": true,
  "annotation_only_after_all_paths_sealed": true,
  "arms": {
    "Layout3OPD": {
      "steps": {
        "0": {
          "m_vIoU": 17.83563305391126,
          "m_sIoU": 40.02151194629846,
          "pseudo_loss": 0.2787807813147083
        },
        "1": {
          "m_vIoU": 17.83730989628893,
          "m_sIoU": 40.02277613616385,
          "pseudo_loss": 0.27829393494175747
        },
        "5": {
          "m_vIoU": 17.836859189497016,
          "m_sIoU": 40.00873748376179,
          "pseudo_loss": 0.2749056311440654
        },
        "15": {
          "m_vIoU": 17.83238321684411,
          "m_sIoU": 39.97566796340591,
          "pseudo_loss": 0.27141935291001573
        },
        "30": {
          "m_vIoU": 17.912638476238733,
          "m_sIoU": 40.01703026324734,
          "pseudo_loss": 0.26026592549169436
        },
        "60": {
          "m_vIoU": 18.511621295270047,
          "m_sIoU": 40.89418928944889,
          "pseudo_loss": 0.2225453495630063
        }
      },
      "pseudo_loss_down_but_dense_down_queries": 7,
      "pseudo_loss_down_but_volume_down_queries": 6
    },
    "Layout3Direct": {
      "steps": {
        "0": {
          "m_vIoU": 17.83563305391126,
          "m_sIoU": 40.02151194629846,
          "pseudo_loss": 0.2787807813147083
        },
        "1": {
          "m_vIoU": 17.837739586175772,
          "m_sIoU": 40.0231899130371,
          "pseudo_loss": 0.27824436634546146
        },
        "5": {
          "m_vIoU": 17.825622738612285,
          "m_sIoU": 39.94571041844783,
          "pseudo_loss": 0.2738833632902242
        },
        "15": {
          "m_vIoU": 17.81862750173031,
          "m_sIoU": 39.89571698005732,
          "pseudo_loss": 0.269728280545678
        },
        "30": {
          "m_vIoU": 17.905736923898953,
          "m_sIoU": 39.98600533438852,
          "pseudo_loss": 0.258951750874985
        },
        "60": {
          "m_vIoU": 18.503029444136352,
          "m_sIoU": 40.37804209386931,
          "pseudo_loss": 0.22376016073394567
        }
      },
      "pseudo_loss_down_but_dense_down_queries": 7,
      "pseudo_loss_down_but_volume_down_queries": 7
    }
  }
}

仅16条机制探针；不按GT选步、不代表128或全量收益。原终点须逐位复现，未修改原已封存结果。
