# R50 full算术／成本／评分合同静态核对

本次只读源码与GT前方法补充锁，没有读取运行中的chunk、case、raw数值、预测或任务GT。没有修改当前运行代码和任何既有锁计划。以下是接口核对，不是对未来封存产物的PASS证书。

|接口|生产者与消费者的确切字段|静态结论|
|---|---|---|
|三臂输出|full_predict50保存 `arm_names`、`scores[3,256]`、`queries[3]`、`selected_original_boxes[3,6]`、`source_boxes/source_scores`；算术审计同名读取，评分计划 `prediction_schema.selected_boxes=selected_original_boxes`|一致；不能把旧R24的`boxes`或`query_indices`字段直接填入此计划。|
|外层封存|full预测seal使用 `n=5912`、`arms=[source,old_G,wideentry_oldW]`；原R8源seal则使用 `n_samples`、`manifest_sha256`、`source_checkpoint_sha256`|两种seal字段有意不同；算术审计已分别读取，不能统一改成`n`。|
|DEV关联|生产者和算术审计均以image/point/meta三路径加完整caption关联；评分另核platform/scene，并用old DEV ID检旧IoU|一致。若保存元数据seed有差异，评分计划必须在GT前填非空 `DEV_seed_mismatch_explanation`，不能假定ID/alias/seed相同。|
|帧绑定|prepared的 `bindings`、证据的 `frame_bindings` 均为image_sha256/point_sha256/mask_cache_sha256；full_visual_observation50正好返回这三项|一致；frame自身另有camera/metadata等字段，不能把整个frame case字典与三项绑定比较。|
|几何数组|`legacy_support/coverage/projection_iou/depth_trim_count/quality/valid`、`old_weights/wide_old_weights`、`old_pool/wide_pool`同名；full额外有`strict_counts`与`mask_counts`两个别名|一致。cheap桥接把两个full别名均对旧DEV的`mask_counts`核验；旧DEV没有`strict_counts`不是缺数据。|
|证据调用成本|CallLedger COUNTS的 `new_input_forwards_all_processes_of_this_chunk`、`logical_requests`、`historical_exact_cache_logical_uses`、`historical_whole_case_logical_uses`、`nonhistorical_logical_uses`；job的`max_new_calls`|独立成本草案字段匹配。物理COMPLETED JSON数不能拿logical_requests替代；chunk间相同输入重复执行仍算不同实际前向。|
|DINO raw|新call的raw记录 `file_PRIVATE/sha256/input_tokens`，NPZ含 `pred_boxes_normalized_cxcywh[900,4]`、`scores[900]`；value含boxes_PRIVATE/scores/query_sha256|一致，可独立原阈值/NMS/top10重放。旧wholecase已unflip，不能再用raw flip路径重复反转。|
|numeric raw|raw含yes/no token IDs、logits、logsumexp、logits_to_keep；value含log_odds_yes_no/yes_no_probability/input_tokens/generated_tokens/image与prompt SHA|一致；stored log-odds须用float32 LSE减法复现，case full-gray仍按原Python float差。|
|实际耗时与显存|COUNT的 `seconds_this_process`、`load_seconds`、`peak_bytes`、`model_unchanged`、runtime.packages|成本草案字段匹配。load/call时间与ledger wall重叠，peak取max，不加总为部署延迟。模型assets仅与受锁manifest/recipe绑定，未另行重新载入权重。|
|评分产物|未来score保存五个成员：ALL5912_TASK_ROWS_PRIVATE、TASK_SUMMARY、ALL_SCENE_SUMMARIES_PRIVATE、ALL_ORDERED_NEGATIVE_COMPARISONS、SCORING_RECEIPT；另有SEALED|新保存标量审计精确要求这五个成员。当前评分器不生成FACTORS或full门裁决，不能从旧DEV字段名假设未来存在。|

## 新增明确门的接线

`experiment/FULL_GATE_AND_TTA_ADVANCE_BEFORE_FULL_GT.json` 是root新写的独立补充锁，不改原运行freeze。它明确候选wideentry_oldW、ALL5912两平台等权mIoU、references=[old_G,source]，每项strict gain>1e-10 pp；没有平台／个例零损害附加门。保存标量审计会核其原freeze摘要，并从已存IoU独立重建门。

`score_full50_draft.py`本身只出评分产物，没有 `full_gate_decision` 文件。未来 `finalize50_draft.py` 的 `plan.full_gate_decision` 可直接指向新重汇总输出 `REGISTERED_FULL_GATE_RECHECK.json`，其 `selected_candidate/full_gate_PASS/full_score_seal_sha256` 与最终报告需求相容，避免期待一个尚不存在的root门文件。

全量评分计划循环对输入路径有较宽的 `'FINAL' not in rel` 检查；因此不要把本轮 `FINAL_REPORT_EXPORT_PLAN_LOCK.json` 等报告／发布文件整批塞进评分输入pins。评分与报告计划保持独立即可，无需改动现有scorer。新门文件名不含该禁词，能按原合同绑定。

评分器顶部导入的独立几何core只有函数定义及数学依赖；已读源码未见输入／GT文件访问。实际cohort、P5索引和GT容器仍在main授权和prediction audit PASS之后读取。若root未来要求连数学模块导入也严格后置，应在评分程序正式锁前单独修订并重签；这不是当前数据schema矛盾，本次没有编辑它。

## 裁决

没有发现需要修改当前运行代码才能解决的核心字段矛盾。仍须等真实seal后按实际hash核验：尤其所有index对应成员、原DEV端点逐位桥接、numpy/BLAS归约与原布局、源seed解释及每chunk完整性。静态兼容不能替代这些执行检查。
