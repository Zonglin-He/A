# TubeDETR BoxFixed＋Global6：选定后完整总体

128开发曝光和全量重叠，全量不再用于参数选择。无OPD/训练/448。Spatial为BoxFixed＋Native，Temporal为Source＋Global6，Full为BoxFixed＋Global6。旧五行消融保持，不将无OPD算法命名WithoutOPD。缓存部署成本不能冒充新端到端成本。

{
  "status": "complete",
  "dataset": "vidstg",
  "queries": 10303,
  "parameters": {
    "gamma": 0.0125,
    "admission": 0.4,
    "guard": 0.4,
    "link": 0.5
  },
  "means": {
    "Source": {
      "v": 13.116927200869647,
      "s": 41.43789381359137,
      "t": 28.67402104763812
    },
    "Spatial": {
      "v": 12.839582864053249,
      "s": 40.858792497215475,
      "t": 28.67402104763812
    },
    "Temporal": {
      "v": 16.26439352769619,
      "s": 41.43789381359137,
      "t": 36.57563023741112
    },
    "Full": {
      "v": 15.969794914375218,
      "s": 40.742112429094945,
      "t": 36.57563023741112
    }
  },
  "unobserved_sIoU": {
    "Source": 41.43038576319615,
    "Spatial": 40.80553233159284,
    "Temporal": 41.43038576319615,
    "Full": 40.6804533467278
  },
  "observed_sIoU": {
    "Source": 42.82628955010402,
    "Spatial": 44.37709960273348,
    "Temporal": 42.82628955010402,
    "Full": 44.77309489643071
  },
  "Full_minus_Temporal_pp": -0.2945986133209698,
  "loss_over5pp_queries": 401,
  "prediction_seal_sha256": "2eaeb2d4b45c1c89423434ea53ff649c8b4ee28e68f58b9b0337411c1cc5aadc",
  "all_predictions_sealed_before_labels": true,
  "mean_admitted_frames": 34.375,
  "mean_changed_from_ERef_frames": 13.234375,
  "not_independent_confirmation": true,
  "paired_difference": {
    "queries": 10303,
    "parents": 732,
    "mean_pp": 2.852867713505575,
    "CI95_pp": [
      2.4023228252762365,
      3.3257742390356055
    ],
    "equal_parent_mean_pp": 3.4282375146656503,
    "wins": 5600,
    "losses": 3058,
    "ties": 1645,
    "below_minus5": 1390
  },
  "equal_parent_Full_m_vIoU": 17.90116583712737,
  "parents": 732
}
