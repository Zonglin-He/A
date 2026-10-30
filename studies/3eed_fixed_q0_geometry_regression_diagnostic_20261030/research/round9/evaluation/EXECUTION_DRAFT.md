正常评分接口已按runner a32166冻结；真实评分仍须完整normal预测/source_cache封存及root明确release。

正常分支评分入口（需要root预先写入真实release SHA）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/absolute/workspace/crossplatform/env/bin/python /absolute/workspace/research/round9/evaluation/score_normal.py \
 --predictions NORMAL_SEALED_DIR --manifest NORMAL_GT_FREE_INPUTS.json \
 --scoring-index R5_SCORING_INDEX_PRIVATE.json --source-checkpoint SOURCE100.pth \
 --release NORMAL_CPU_SCORE_RELEASE.json --release-sha256 RELEASE_SHA256 \
 --program-root /absolute/workspace --output NEW_NORMAL_SCORED_DIR
```

release字段：CPU_SCORE_RELEASE=true,complete_normal_predictions_sealed_before_GT=true,stage=normal,source_model_source_only_Waymo=true；绑定prediction_seal_sha256、manifest_sha256、scoring_index_sha256、source_checkpoint_sha256、scorer_program_sha256、geometry_metrics_sha256，以及geometry_dependencies_sha256精确两项（见CPU_SCORE_RELEASE_TEMPLATE.json）。

normal NPZ需固定ARMS顺序与11query全q0，source/arm raw+eval6、source scores256/q_source、arm_query_index、arm_status、2×6 pseudo_target6和pseudo_reliability6。预测目录完整SEALED需n_arms=11,n_samples,files,program_sha256,program_sha256_base=workspace_root,strict_FP32_flags,manifest_sha256,source_checkpoint_sha256。

normal predictions gate验证全部文件/代码/source/manifest；随后全例NPZ契约验证；两者通过后才加载评分index/metadata GT。失效必须显式失败，不跳样本。当前CLI/字段为接口draft，实际runner schema确认之前不是执行锁。

最终scorer SHA d6e883085bd095eddeca51eab67aee3115b37ff8b6bb10a397a834d759c17d86。入口仍为上方CLI，normal source-cache已逐例独立校验。全部GTindex/sidecar仅在root释放后生成，禁现在调用。

无GT normal审计（先等待真实SEALED与root通知）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/absolute/workspace/crossplatform/env/bin/python /absolute/workspace/research/round9/evaluation/audit_normal_unlabelled.py \
 --predictions /absolute/workspace/crossplatform/round9/normal160_attempt002 \
 --manifest /absolute/workspace/research/round9/experiment/diagnostic160_gpu_inputs.json \
 --program-root /absolute/workspace --source-checkpoint SOURCE100.pth \
 --expected-seal-sha256 ACTUAL_NORMAL_SEAL_SHA --output NEW_UNLABELLED_AUDIT.json
```

GT13评分（另一份root release，stage=GT_controlled_NOT_TTA；正常评分先生成GTindex并seal，GT分支运行与seal完成后才能评分）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/absolute/workspace/crossplatform/env/bin/python /absolute/workspace/research/round9/evaluation/score_gt_controlled.py \
 --predictions GTCONTROLLED_SEALED_DIR --normal-scored NORMAL_SCORED_DIR \
 --release GTCONTROLLED_CPU_SCORE_RELEASE.json --release-sha256 RELEASE_SHA256 \
 --source-checkpoint SOURCE100.pth --program-root /absolute/workspace --output NEW_GTCONTROLLED_SCORED_DIR
```

GT评分release绑定GT_prediction_seal_sha256、normal_scored_seal_sha256、source_checkpoint_sha256、scorer_program_sha256、capacity_metrics_sha256、geometry_metrics_sha256、normal_scorer_program_sha256、geometry_dependencies_sha256，并要求normal_sealed_before_GT_injection=true。全部正常GTindex/侧车和oracle输出hash校验在读取派生数值GT之前。
