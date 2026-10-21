# 通用轨迹接口：TubeDETR初试

固定原输入哈希16条，时间固定Strong。SourceStrong是原生Source空间+固定Strong时间，未使用旧E_Ref输出修正。所有候选真实执行decoder LayerNorm affine；普通学生视觉特征传播、奖励、梯度子空间和分量更新独立于模型query结构。八个证据分量不是已知不同身份。Direct同奖励同U和约束，8起点2步，仅初试未公平调参，计算不等额。完整预测封存后独立CPU评分，不写主表，不自动全量。

{
  "status": "complete_interface_and_finite_search_trial",
  "queries": 16,
  "metrics": {
    "SourceStrong": {
      "v": 18.401566273370452,
      "s": 41.04339179624309,
      "t": 35.75999052724047
    },
    "CandidateRerank": {
      "v": 16.140793365608246,
      "s": 37.154868792038286,
      "t": 35.75999052724047
    },
    "PolicyCenter": {
      "v": 17.888129462714925,
      "s": 40.14612347525085,
      "t": 35.75999052724047
    },
    "MultiStartDirect": {
      "v": 17.61233415705891,
      "s": 39.92301406552924,
      "t": 35.75999052724047
    }
  },
  "mean_subspace_rank": 8.0,
  "mean_trajectory_clusters": 8.1875,
  "mean_pairwise_trajectory_IoU": 0.7844754319858995,
  "mean_evidence_confidence": 0.04593032971024513,
  "mean_unobserved_confidence": 0.040971324138808995,
  "candidate_reward_pair_accuracy": 0.45934065934065926,
  "candidate_reward_spearman": -0.10199800199800203,
  "candidate_oracle_m_vIoU": 20.875655033565394,
  "parameter_count": 9728,
  "feature_interfaces": [
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        24,
        14
      ],
      "canvas": [
        373,
        210
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        24,
        14
      ],
      "canvas": [
        372,
        210
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        24
      ],
      "canvas": [
        210,
        373
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        24
      ],
      "canvas": [
        210,
        373
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    },
    {
      "feature": "ordinary ResNet layer3 before text fusion",
      "channels": 1024,
      "spatial_stride": 16,
      "grid": [
        14,
        19
      ],
      "canvas": [
        224,
        298
      ],
      "coordinate_convention": "stride16 raster cells intersected with valid transformed image; not pixel-accurate receptive-field boundaries"
    }
  ],
  "frozen_Strong_time": true,
  "no_new_teacher": true,
  "actual_parameter_executions": true,
  "no_query_residual_or_ASA": true,
  "not_full_population_or_independent_confirmation": true,
  "Direct_not_tuned": true,
  "policy_is_reward_weighted_parameter_search_not_claimed_new_distillation": true,
  "differences": {
    "PolicyCenter-CandidateRerank": {
      "queries": 16,
      "parents": 16,
      "mean_pp": 1.747336097106678,
      "CI95_pp": [
        0.6472514987407844,
        3.0065621322523612
      ],
      "equal_parent_mean_pp": 1.7473360971066776,
      "wins": 9,
      "losses": 4,
      "ties": 3,
      "below_minus5": 0
    },
    "PolicyCenter-MultiStartDirect": {
      "queries": 16,
      "parents": 16,
      "mean_pp": 0.27579530565601595,
      "CI95_pp": [
        -0.29299293706512985,
        0.9239447988903291
      ],
      "equal_parent_mean_pp": 0.2757953056560159,
      "wins": 7,
      "losses": 6,
      "ties": 3,
      "below_minus5": 0
    },
    "MultiStartDirect-SourceStrong": {
      "queries": 16,
      "parents": 16,
      "mean_pp": -0.789232116311538,
      "CI95_pp": [
        -1.5800257449197772,
        -0.04028289702865538
      ],
      "equal_parent_mean_pp": -0.789232116311538,
      "wins": 4,
      "losses": 9,
      "ties": 3,
      "below_minus5": 0
    }
  },
  "execution_cost": {
    "common_basis_backwards": 8,
    "initial_candidate_forwards": 20,
    "policy_center_forwards": 8,
    "Direct_forwards": 24,
    "Direct_backwards": 16,
    "not_equal_compute": true,
    "mean_seconds": 1.4713059730129316,
    "peak_GPU_bytes": 3781681152
  }
}
