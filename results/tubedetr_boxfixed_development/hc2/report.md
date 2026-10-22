# TubeDETR BoxFixed＋Global6：选定后完整总体

128开发曝光和全量重叠，全量不再用于参数选择。无OPD/训练/448。Spatial为BoxFixed＋Native，Temporal为Source＋Global6，Full为BoxFixed＋Global6。旧五行消融保持，不将无OPD算法命名WithoutOPD。缓存部署成本不能冒充新端到端成本。

{
  "status": "complete",
  "dataset": "hc2",
  "queries": 1902,
  "parameters": {
    "gamma": 0.025,
    "admission": 0.4,
    "guard": 0.4,
    "link": 0.5
  },
  "means": {
    "Source": {
      "v": 15.425086700559776,
      "s": 55.89458542601735,
      "t": 26.800738955102155
    },
    "Spatial": {
      "v": 15.40171160524432,
      "s": 55.890902029945465,
      "t": 26.800738955102155
    },
    "Temporal": {
      "v": 25.78437474216269,
      "s": 55.89458542601735,
      "t": 45.58281600816047
    },
    "Full": {
      "v": 25.821066974628685,
      "s": 56.01671909283422,
      "t": 45.58281600816047
    }
  },
  "unobserved_sIoU": {
    "Source": 55.876569165426396,
    "Spatial": 55.8441882265727,
    "Temporal": 55.876569165426396,
    "Full": 55.96462911186908
  },
  "observed_sIoU": {
    "Source": 57.05074663715837,
    "Spatial": 58.47923947042813,
    "Temporal": 57.05074663715837,
    "Full": 59.21769371176884
  },
  "Full_minus_Temporal_pp": 0.03669223246599257,
  "loss_over5pp_queries": 93,
  "prediction_seal_sha256": "d91993d84c5dabb7f4a938ad6de7a9d968ec9bcd7e4bbcf52c3e8b48fc1f8f01",
  "all_predictions_sealed_before_labels": true,
  "mean_admitted_frames": 39.5703125,
  "mean_changed_from_ERef_frames": 13.09375,
  "not_independent_confirmation": true,
  "paired_difference": {
    "queries": 1902,
    "parents": 227,
    "mean_pp": 10.395980274068913,
    "CI95_pp": [
      9.55979434437959,
      11.179646758185411
    ],
    "equal_parent_mean_pp": 10.252353759404176,
    "wins": 1423,
    "losses": 421,
    "ties": 58,
    "below_minus5": 233
  },
  "equal_parent_Full_m_vIoU": 26.212817956132906,
  "parents": 227
}
