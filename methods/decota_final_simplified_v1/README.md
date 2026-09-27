# DeCoTA：已选定 C+D 方法的精简实现

这是用户于 2026-09-15 确认设计后整理的**同算法工程实现**，不是另一轮方法搜索。
数学参照为 `artifacts/decota_final_simplification_v1/FINAL_CONFIG.json`，SHA-256：
`ed8f7d31e9a810fa66a2ee714667bce566cb203dab8a6016f96fcce39dcbf12e`。

本包不导入 `vg_tta` 或 `scripts`。没有历史 baseline、oracle、调参、评价标签、corruption、
插值输出、自动任务或全量队列。TA-STVG 官方网络仍来自 `external/TA-STVG`；
Grounding DINO-T 仍通过已安装 Transformers 和原本地权重加载。

## 唯一流程

完整视频与查询 → 两个原生 offset → 匹配 FP32 原生预测 →
原生区间内均匀四时刻 / 单视图冻结 DINO-T / S2 →
空间 query+三个实际 LN 的 10 步参数适应 / best →
全帧原生 actionness 的 hard structured teacher →
原生时间头的 5 步参数适应 / last → 参数位移乘 0.25 →
两组真实状态插回原模型完整推理 → 原生双 offset 解码 → episode 重置。

| 固定项 | VidSTG → HC-STVG1 | HC-STVG2 → VidSTG |
|---|---:|---:|
| 空间学习率 / Adam eps | .005 / 1e-8 | .05 / 1e-8 |
| 空间步数 / 选择 | 10 / best，含初态 | 10 / best，含初态 |
| 空间 loss | 接受锚点的 raw(5L1+2GIoU) 总和 / planned 4 | 相同 |
| 时间学习率 / AdamW eps | .001 / 1e-4 | .1 / 1e-4 |
| 时间步数 / 选择 | 5 / last | 5 / last |
| actionness median 中心系数 α | 1 | .5 |
| 教师 native posterior 权重 λ | .1 | .1 |
| 时间 loss | NLL + mode margin(.2)，训练 KL 系数为 0 | 相同 |
| 最终参数位移系数 η | .25 | .25 |

空间 1,792 参数；时间 66,306 参数；总计 68,098。无 weight decay、无回溯。
无接受锚点时只跳过空间优化；时间照常运行。非有限更新恢复对应初态并记录，不伪造成功。
没有按测试视频选择 η；不因新时间区间重新选择专家帧。

## Python 入口

从项目根目录运行；当前环境使用 `.conda/tubedetr/bin/python`。

```python
from methods.decota_final_simplified_v1 import DeCoTAPredictor

predictor = DeCoTAPredictor.from_pretrained("vid_to_hc1")
result = predictor.predict(
    frames,     # numpy.uint8，RGB [T,H,W,3]，不是 BGR
    frame_ids,  # 真实、严格递增的原始 frame IDs，至少四位置
    {"caption": query, "index": "sample", "width": W, "height": H},
)
boxes = result["boxes"]       # 完整采样网格上的 normalized cxcywh
interval = result["physical_interval"]  # [start_frame, end_frame)，右端排他
```

`.indices` 是采样网格的闭区间；`.physical_interval` 是原始帧号的半开区间。
这两种定义不能混用。元数据字段采用白名单，GT 框/时间/标签不允许进入在线接口。
一实例同一时刻只允许一个 episode。源 checkpoint 逐 episode 恢复，不能跨视频复用优化器。

完整审计：`DeCoTAPredictor.from_pretrained(direction, audit=True)`。
它增加状态哈希、优化轨迹和输入回执，**不改变计算目标或所选参数**。
工程排查可设 `cached_replay=False` 重算冻结后缀；它不是另一套方法超参数。

## 单次命令行

NPZ 只需包含 `frames` 和 `frame_ids`，不允许 object pickle。
输出必须为新文件，避免覆盖已有实验。

```bash
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/predict_decota.py \
  --direction vid_to_hc1 \
  --frames /absolute/path/input_frames.npz \
  --query "the person opens the bag" \
  --output /absolute/path/prediction.pt
```

该入口不自动采样视频：由调用方显式提供与原实验一致的物理采样网格，避免改变采样协议。
运行依赖现有两个 checkpoint、DINO snapshot、RoBERTa/Video-Swin/ResNet 缓存、
Stanza 资源及 `artifacts/tastvg_runtime` 的官方模型构造兼容目录。

## 文件职责

| 文件 | 唯一职责 |
|---|---|
| `config.py` | 两方向明确的最终配置，不继承旧消融 override |
| `predictor.py` | 单视频协调、观测、两组参数适应与最终输出 |
| `objectives.py` | 两个固定 loss、合法区间投影、原生解码 |
| `optim.py` | 普通 Adam/AdamW、空间 best / 时间 last、η 收缩 |
| `replay.py` | 保持实际梯度的冻结前缀复用 |
| `observations.py` | 自动 query parse、四帧选取、冻结 DINO-T / S2 |
| `backbone.py` | 输入、两个 offset、捕获与完整原模型状态插回 |
| `loader.py` / `_tastvg_load.py` | 严格源架构和 checkpoint 兼容加载 |
| `tensors.py` | 状态复制与哈希等小工具 |
| `release.py` / `RELEASE.json` | 整理版代码与官方依赖的完整性校验 |

`SOURCE_MAP.json` 记录提取来源；源文件保持不动。模型加载使用 checkpoint 的 `model_ema`
（存在时），不把同文件的非 EMA `model` 悄悄替换成部署权重。

## 做了哪些等价提速

1. 每个 episode 只捕获一次原生 FP32 时间头输入，不再另建完整后缀重复捕获。
2. 空间迭代仅重算第二遍实际 PosDecoder；缓存其不依赖空间参数的输入。
   **六个空间 block、query residual 的完整梯度和每步迭代框引用仍全部重算**。
3. 锚点 tensor、合法区间索引只构造一次；不再每步重复传输。
4. 只保存选中状态，完整轨迹通过 `audit=True` 开启；不再每步强制 CPU 拷贝与 SHA。
5. 插回原模型时只备份会写入的参数，不复制整个冻结 decoder。
6. 同一 episode 的相同文本解析复用，不合并大小写不同的解析请求。
7. 默认只输出主方法，不计算八个研究臂和五点 η 曲线。

仍保留最终完整原模型前向及逐值一致性检查。不改变数值精度、Adam 语义、NMS、阈值、
候选截断顺序、采样/图像预处理、时间 cell、MAP tie-breaking、损失权重或步数。

## 核验与边界

见 `artifacts/decota_runtime_cleanup_v1/REPORT.md`：CPU 数学与异常恢复测试、
64 条既有开发缓存的 GPU 时间轨迹、两源 EMA 空间图/梯度检查、
两条真实视频八位置完整集成检查。所有已比对的预测与参数采用精确相等，不放宽到指标接近。

这些检查支持本次工程等价性，**不是新准确率实验，也不是全量数据已经由新执行器重跑**。
正在执行的 VidSTG 全量仍保持旧锁定执行器和两个 CURRENT 注册文件不变，以免破坏其哈希审计。
方法设计已按用户选择固定；整理版的可调用 API 已就绪，但本次不热切换正在执行的作业。
历史生产版本仍由 `methods/CURRENT_METHOD.json` 表示，不能把此包的创建写成注册已经改变。
