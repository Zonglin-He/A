R10 仅执行 direct geometry 评分，尚未授权真实目标评分。实际四臂名固定为：`source`、`full_rgb`、`no_rgb_coordinate`、`rgb_gated_pointmask_ablation`。源捕获已有无GT独立审计；G代码/config/实际wrapper完整审计通过后再冻结最终评价锁。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/absolute/workspace/crossplatform/env/bin/python /absolute/workspace/research/round10/evaluation/score_geometry.py \
 --predictions COMPLETE_GEOMETRY_PREDICTIONS_DIR \
 --source /absolute/workspace/crossplatform/round10/source160_attempt001 \
 --manifest /absolute/workspace/crossplatform/round10/source160_attempt001/manifest_input.json \
 --scoring-index ROOT_BOUND_R5_SCORING_INDEX_PRIVATE.json \
 --source-checkpoint SOURCE100_CHECKPOINT.pth \
 --release ROOT_R10_CPU_SCORE_RELEASE.json --release-sha256 RELEASE_SHA256 \
 --program-root /absolute/workspace --output NEW_R10_SCORED_DIR
```

入口先核root release、全部source/G文件及代码hash和所有例子的NPZ契约，再打开独立评分index/metadata。R10队列排除R9全部IDs/pointpaths，不复用旧GT侧车或旧source身份标签。仅后置读取target bbox，通过独立first7/DroneZshift变换；不读取class或positive-map，不做GT候选重选/PCA/GTyaw干预。

输出包含all、full/noRGBcoord共同union_changed、common_eligible、pointmask pair共同集合，各集合对source/全部臂使用同一分母；各臂own_changed仅描述。按平台报告IoU/Acc25/50、中心/尺寸误差、signed gain/harm/zero和scene等权均值/描述性区间。尺寸误差是GT局部尺寸参数误差，不等价yaw0全局包络。

后续跨输入更新只有在伪框质量的预锁门槛通过后才另立计划。A/B需先按原rawindex分、后replacement采样，不能把同一原始点在两侧的重复数组行当独立监督。当前不运行TTA或额外模型。
