# 参数在哪改、实际影响什么

新工作区每个 JSON 的 `kind` 决定允许哪些字段。所有字段在 [`desta3d/config.py`](../../desta3d/config.py) 中有带类型的默认值，`inspect` 会输出最终生效配置。

## Predictor / cached fit

预设：[Local](../../configs/desta3d/direction_local.json)、[GlobalMean](../../configs/desta3d/direction_global_mean.json)、[R16](../../configs/desta3d/direction_r16.json)。

| JSON 字段 | 默认 | 改变什么 | 实际消费位置 |
|---|---:|---|---|
| `model.feature_dim` | 128 | cache 中 z/qT/qS 的真实宽度；不能拿它当网络容量旋钮 | `DirectionMixer.__init__/forward` |
| `model.hidden_dim` | 128 | Linear/DWConv3D 内部宽度，不改输入表示 | `DirectionMixer.__init__` |
| `model.output_rank` | 256 / 16 | coefficient 通道数，必须等于冻结 basis 列数 | `DirectionMixer.__init__` |
| `model.global_mean` | false | `h0 + local + mean_THW(h0)`，不加参数 | `DirectionMixer.forward` |
| `model.output_init_std` | .001 | 输出 Linear 小非零初始化；零输出没有定义良好的归一方向 | `DirectionMixer.__init__` |
| `model.radius` | .13545580427763146 | `project()` 后 residual 相对原始 F 的范数；**不进入 fit loss** | `DirectionMixer.project` |
| `training.seed` | 20260928 | 初始化和 Python query shuffle；不是按成绩换 seed | `_fit` |
| `training.steps` | 200 | actual Adam steps；固定末态 | `_fit` |
| `training.batch_size` | 4 | 每步等权 query 累积数；各 query THW 可不同 | `_fit` |
| `training.learning_rate` | .001 | AdamW 学习率 | `_fit` |
| `training.weight_decay` | 0 | AdamW weight decay | `_fit` |
| `training.clip_norm` | 1 | 全 trainable 参数梯度 norm clip；日志记录 clip 前值 | `_fit` |
| `training.device` | cuda:0 | 显式 `fit-cache` 时才使用；`summary` 永远 CPU | `_fit` |
| `training.threads` | 4 | Torch CPU 线程数 | `_fit` |
| `training.minimum_free_gib` | 8 | 工程磁盘底线，不接受 <8 | `_fit` |

Loss 固定为每 query 完整 field 的 `1-cos`，不混 CE、KL、幅度或 gate。R16 target 先在 CPU FP64 投影再转 FP32 训练；这里的终态 R16 cosine也针对这个训练 target，不冒称旧报告的 FP64 未舍入 target 读数。

`cache.root`、`basis_file`、`channel_basis_file` 是相对 workspace 的路径或绝对路径；对应 SHA 在读取前校验。修改内部宽度不需要改变这些字段。修改数据或 basis 必须指向你另行准备、独立封存的新输入，不能只把旧文件的 hash 改掉来绕过验证。

## 3D adapter

预设：[frozen B1 结构](../../configs/desta3d/adapter_b1.json)、[FiLM/LN TTA scope](../../configs/desta3d/adapter_tta.json)。

| 字段/符号 | 默认 | 说明 |
|---|---|---|
| `in_channels`, `query_dim` | 2560 / 2560 | PTD4B merger/query 原始宽度 |
| `hidden_dim` | 128 | adapter 的 latent 宽度；改变它会使 B1 checkpoint 不兼容 |
| `architecture` | dual3d | 可选 shared3d、early_factorized；参数共享方式变化 |
| `p1_enabled` | false | 真实时间差分 + temporal dilation 组，属于机制改变 |
| `train_stage` | frozen | A evidence / B integration / tta / frozen |
| `freeze_tta_gates` | true | tta 时只开 FiLM/LN：66,816；false 则包括两 gate：66,818 |
| `gate_event`, `gate_spatial` | 构造初始 logit −6 | forward 使用 sigmoid；**已加载 B1 的值由 checkpoint 决定** |
| `gate_override` | None | 原 adapter forward 支持 scalar 或 `(spatial,event)` 直接 gate 值；不是 logit |
| `QueryFiLM` gamma 系数 | .1 | `gamma=.1*tanh(raw_gamma)`；位于原模型代码，未伪装成新 JSON 开关 |
| ChannelOnlyLayerNorm epsilon | 1e−5 | 每 THW 位置只沿 channel 归一化 |

原模型的可训练范围见 `parameter_groups()` 与 `set_train_stage()`。新 factory 只显式处理 tta gate 冻结，不重写 adapter。不要用 `strict=False` 吞掉宽度/架构不匹配，再声称从同 B1 开始。

## Pixel privilege / external evidence

[配置](../../configs/desta3d/pixel_views.json)通过 `make_pixel_views` 真正传给原 `build_views`。

| 字段 | 默认 | 意义 |
|---|---:|---|
| `views.temporal_dim` | .25 | 预测 interval 外的 RGB 强度乘数；1 为时间 no-op |
| `views.spatial_blur_radius` | 8 | box 外 Gaussian blur 像素半径；0 为模糊 no-op |

保留帧 ID、顺序、像素尺寸、坐标系；T 后 S 产生 TS。局部无效框保持原图，physical-time 插值和 duplicate 聚合按原 parser 合同。配置不修改 teacher prompt、tokenizer、decode recipe 或 loader；这些有独立协议。

## 旧 v2 TTA / source fit：只读导航

这些参数仍在原文件中，不由新的 cached-fit JSON 控制：

| 想改的内容 | 原实现 |
|---|---|
| latent/ref/event/alignment/parameter anchor 权重 | [`CalibrationWeights`](../../vg_tta/desta3d_v2_tta_objective.py)，默认 1/1/1/0/1；旧 calib-align 用 alignment=.01 |
| calibration 3步、lr1e−5、wd0、clip1 | [`adapt`](../../vg_tta/desta3d_v2_tta_pilot.py)；其中 steps/lr 是函数参数，wd/clip 在函数体 |
| 固定 brightness1.05 / contrast.95 | [`make_mild_photometric_view`](../../scripts/desta3d_tta_run_v1.py) |
| source 两阶段 optimizer / loss / schedule | [`desta3d_v2_source_fit.py`](../../scripts/desta3d_v2_source_fit.py) 与对应 protocol |
| native endpoint 与 full-vocab coordinate action support | [`desta3d_v3_actuation_full_vocab.py`](../../vg_tta/desta3d_v3_actuation_full_vocab.py) |

改旧研究 runner 的常数会触发旧 LOCK/hash 检查。历史复现保持原文件；手动实验在新目录登记实际参数和调用链。
