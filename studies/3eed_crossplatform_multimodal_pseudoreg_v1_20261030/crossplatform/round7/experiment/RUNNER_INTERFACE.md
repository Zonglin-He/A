# R7v1真实跨域runner：根代理注册并启动

`run_crossdomain.py`仅由根代理在新计划中授予GPU后执行；脚本自身不调度GPU，外层须持有全项目同一 `research/round2/gpu.lock`。smoke/full共用代码；smoke只做无标签工程审计，完整5912条预测封存前不进行目标GT评分。代码不读取目标类别、positive_map或数值GT；模型只看到白名单XYZ和纯文本。相机允许字段与RGB由G使用，原annotation容器通过camera helper流式解析但仅保留camera字段。

## 执行和计划

```
crossplatform/env/bin/python -B crossplatform/round7/experiment/run_crossdomain.py \
  --phase smoke --manifest <smoke_gpu_inputs.json> \
  --plan <root-locked-plan.json> --plan-sha256 <sha256> --output <fresh-output>
```

全量仅将`--phase full`及相应manifest/新计划/输出更换。不能覆盖已有目录；失败保留部分文件与日志，不写SEALED。根代理另锁新attempt，不将部分输出称完成。

根计划必须有以下字段：

- `phase`, `GPU_execution_authorized:true`, `batch_size:1`, `expected_samples`, `manifest_sha256`。
- `program_sha256`：相对于`crossplatform/round7`的代码路径→SHA，须包含runner/helpers、冻结adapter/G/camera及其依赖；runner逐项复核。
- `r6_factory_sha256`、`upstream_code_sha256`、`roberta_sha256`。调用R6原factory.model，不调用R6configure/dataset。新runtime仅在R7实验目录下链接generic RoBERTa。
- `source_checkpoint_path`, `source_checkpoint_sha256`（必须是`0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd`）、`source_training_plan_sha256`, `source_model_state_fingerprint`。加载`weights_only=True`、严格完整state，验证epoch100、trained_platforms=['waymo']及无联合任务初始化。
- `alpha:.25`, `beta_metres:.1`, `steps:[1,3]`，`G_config_sha256`。
- `missing_distortion_policy:'assume_zero_declared'`，`noun_fallback:'all_content_tokens'|'fail_closed_source_only'`；样本是否fallback逐条留存。输入审计目前5912全部命中固定词典首名词，fallback并未被真实样本需要。
- `seed`, `cudnn_benchmark`, `cudnn_deterministic`, `sentinel`（与完整描述组成精确模型输入）。

输入manifest严格9keys：`id/platform/scene/pointpath/metapath/imagepath/caption/suffix/seed`，平台仅Drone/Quad。id稳定关联根代理的独立评分index，frame/objectidx/GT只能留在评分私有index。此处DQ suffix=None可接受，投影不使用Waymo suffix。

采样逐字节采用R5约定：raw float32，`default_rng(seed).choice(N,16384,replace=N<16384)`，Drone在float32 copy上原位z+=1.8，Quad不变。记录原文件/采样点/indices/image/meta SHA。P是全模型采样输入，不按GT或source框挑点。RGB使用原JPEG decode，不做另一个模型的输入预处理。

## 过程及隔离

每例一次source eval/no_grad前向，由冻结adapter捕获最终两Conv输入和source框；内部Conv重放必须bitexact。纯文本BBF保留真实全部T-token softmax分母。固定q为argmax第一索引；原生GPU argsort Top10单独保存，若ties导致nativeTop1≠q，作为排序口径差异记录，不替换固定q。

每例根据无标签source特征计算

`lr = alpha * beta / max(||h_center(q)||²+1, ||h_size(q)||²+1)`。

在SmoothL1平方区，当前query最大收缩因子≤alpha=.25（可靠性归一化还会减小它）。这是归一化步长，不是把G的位移上限冒充SGD位移裁剪。每个mode分别从同一source重置执行1和3步；没有teacher/EMA、跨样本记忆、backbone或BN更新。G尺寸权重恒0，不回归yaw。

raw source尺寸无效、unsupported noun按注册policy或相机缺失/拒绝时，保留样本，所有臂source fallback。G全部权重0时保留其原因及空优化轨迹。w>0但伪框恰等source可能有零loss/gradient/delta，应按实际变化统计而不是按gate接受数量冒称更新。

sample0另做一次source前向，验证source boxes/scores重复bitexact、真实Conv桥及全部adapt后完整模型指纹/buffers恢复。最后再核全模型指纹、buffers、所有grad=None、requires_grad=False及eval模式。冻结模块已在每次episode恢复参数值/gradient引用/modes/buffers/RNG。

## NPZ与封存

每个`<id>.npz`包含：

- `sample_id`字符串标量。
- `source_boxes_raw6/source_boxes_eval6 [256,6]` float32，`source_text_scores256 [256]`，`source_native_top10 [10]` int64。
- `fixed_query_index`标量，`native_top1_differs_fixed_query`布尔标量。
- `arm_names/arm_modes/arm_status [16]`字符串；`arm_boxes_raw6/arm_boxes_eval6 [16,6]` float32。
- `mode_names [3]`，`pseudo_target6/pseudo_reliability6 [3,6]`，size严格保留source且w_size=0。
- `point_sample_sha256/image_file_sha256 [32]` uint8。

16臂顺序：source；mode依次`rgb_text,pc_only,shuffled_color`，每mode依次`<mode>__direct_G,__param1,__param3,__outputspace1,__outputspace3`。eval尺寸统一clamp至1e-6，其余坐标不变，所有yaw0。源Top1主metric固定q，Top5/10覆盖另依实际native排名诊断。

`trace.json/jsonl`保留每例mode G诊断、weights支持、1/3步loss、六维梯度、参数梯度norm、nonzero梯度参数数、parameter_delta、featureNorm、spillover、参数/框空间重放误差；另有实际LR、采样/相机证据、forward与端到端时间、allocated峰值。这里的forward时间包含source snapshot/Conv bridge，不能冒称纯模型kernel时间。

`SEALED.json`的`files`包含全部NPZ、trace、重复/恢复控制、注册计划和GPU输入manifest的相对路径→SHA；还绑定source checkpoint/model指纹、manifest/plan/code/upstream hashes及n_samples。只有根代理验收此seal并发布独立CPU_SCORE_RELEASE后，评分进程才可读取目标数值GT。runner自身没有评分入口。

## 当前检查范围

`check_runner_cpu.py`仅AST解析及临时合成采样/NPZ roundtrip：核R5 float32 Drone变换、9key白名单拒绝GT额外字段、16臂schema和raw/eval clamp。未构造或加载真实模型，也未GPU/GT评分。模型算子、严格checkpoint/Conv桥、真实颜色支持率、源恢复及实际时间必须由真实smoke无标签审计确认。
