# DESTA-3D：手动读代码与调参入口

先读本页，然后打开 [参数表](HYPERPARAMETERS.md) 和 [计算图/文件地图](CODE_MAP.md)。

## 最常用的五个文件

| 想看什么 | 打开哪里 |
|---|---|
| 共享 THW、QueryPool、FiLM、双 reader、残差 gate | [`vg_tta/desta3d_v2.py`](../../vg_tta/desta3d_v2.py) 的 `Desta3DAdapterV2.forward` |
| 可直接编辑的参数 | [`configs/desta3d/`](../../configs/desta3d/) 下的 JSON |
| 整理后的 coefficient predictor 前向、归一化和 loss | [`desta3d/models.py`](../../desta3d/models.py) |
| 手动 cached training 的完整循环 | [`desta3d/train_cached.py`](../../desta3d/train_cached.py) 的 `_fit` |
| PTD 真正接收哪个 tensor、两遍 native decode | [`vg_tta/desta3d_v2_ptd.py`](../../vg_tta/desta3d_v2_ptd.py) |

`desta3d/` 是本次新增的可编辑工作区。历史 `scripts/`、`vg_tta/`、`protocols/` 和 sealed artifact 保留原样，原有 hash 锁不会因整理失效。新入口不导入历史实验 launcher，也不自动启动任务。

## 第一步：看有效配置与模型结构

在仓库根目录使用本地已有 PTD Python 环境：

```bash
.venv-ptd-audit/bin/python -B -m desta3d inspect --config configs/desta3d/direction_local.json
.venv-ptd-audit/bin/python -B -m desta3d summary --config configs/desta3d/direction_local.json
.venv-ptd-audit/bin/python -B -m desta3d summary --config configs/desta3d/adapter_tta.json
```

`inspect` 只解析 JSON，连 Torch 都不加载。拼错参数名、把字符串 `"false"` 当布尔值、负学习率等会报错。

`summary` 在 CPU 构造随机初始化结构，列出模块、参数量及可训练名称。它不加载 B1/PTD checkpoint，不读取视频、cache 或标签。查看 B1 结构的配置名 `adapter_b1.json` **不代表已经加载 B1 权重**。

## 第二步：复制一份配置，改自己的参数

```bash
cp configs/desta3d/direction_local.json configs/desta3d/my_direction.json
```

优先看这些字段：

```json
{
  "model": {
    "feature_dim": 128,
    "hidden_dim": 128,
    "output_rank": 256,
    "global_mean": false,
    "radius": 0.13545580427763146,
    "output_init_std": 0.001
  },
  "training": {
    "steps": 200,
    "batch_size": 4,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "clip_norm": 1.0,
    "device": "cuda:0"
  }
}
```

这是节选，实际运行应复制完整 preset。`feature_dim` 是 cache 的真实维数；加宽内部网络只改 `hidden_dim`。`radius` 只影响 residual 投影，不影响 cached direction-only loss，因此修改它不会改变这个训练循环的拟合结果。

## 第三步：只有你显式运行，才开始 cached fit

```bash
.venv-ptd-audit/bin/python -B -m desta3d fit-cache \
  --config configs/desta3d/my_direction.json \
  --run-name manual_local_001
```

输出固定进入 `artifacts/manual_desta3d/manual_local_001/`；同名目录存在就报错，不覆盖旧记录。每次都从配置指定 seed 初始化，固定终态，不自动选 best-step、恢复训练或接 native gate。

该命令只消费既有 **source-GT privileged cached tensors**，只训练 predictor；不加载 PTD，不生成视频预测。它提供训练代码供你手动检查，**不是恢复已停止的 offline predictor 研究队列**。

| 输出 | 用途 |
|---|---|
| `CONFIG.json` | 所有默认值展开后的实际配置 |
| `INPUTS.json` | cache/basis/code SHA、版本、数量及 GT privilege 标记 |
| `INITIAL.pt`, `FINAL.pt` | 初态、终态、整数 key 的 Adam state、实际 step |
| `HISTORY.jsonl` | 每步 query index、loss、clip 前 norm、Adam counters |
| `terminal_coefficients/` | 全部 train/dev 终态输出；可独立复算 |
| `REPORT.json` | NumPy FP64 cosine mean/median/count、梯度与 clip 统计 |
| `SEAL.json`, `COMPLETE.json` | 输出 hash 与完成标志 |
| `RECEIPT.json`, `FAILURE.json`（若失败） | 本次实际 wall time 与保留的错误 |

使用 `cuda:0` 时先拒绝已有 CUDA compute process，并以文件锁串行化本工作区 fit；每步检查 8 GiB 磁盘底线。完整终态输出所需空间会先估算。手动运行的耗时保存于本次 receipt，**不会自动混入旧研究总账**。

公开 GitHub 不含 cache、视频、标签或权重。公开 clone 若需要使用本机已有缓存，可传 `--workspace '/home/wwww/visual grounding'`；运行输出也会进入该 workspace 的 `artifacts/manual_desta3d/`。preset 内 cache/basis hash 固定且必验，读的是训练缓存，不会去找 fresh388 或其他样本。

## Python 里直接调模型

```python
import torch
from desta3d.config import AdapterConfig, DirectionConfig
from desta3d.models import build_adapter, DirectionMixer, direction_loss

# 复用原始 adapter 实现，默认冻结；加载自己明确选择的 checkpoint。
adapter = build_adapter(AdapterConfig())
# adapter.load_state_dict(checkpoint["adapter"], strict=True)

# basis: [visual_dim, rank]，必须真实正交且与缓存/目标空间一致。
predictor = DirectionMixer(basis, DirectionConfig(hidden_dim=256))
coeff = predictor(cache)                 # [B,T,H,W,rank]
loss = direction_loss(coeff, oracle)     # 每 query 整个 field 的 1-cos
delta = predictor.project(coeff, norms)  # ||delta_q|| = radius * ||F_q||
```

`cache` 的字段、坐标与 loss 定义见 [CODE_MAP](CODE_MAP.md)。使用 rank16 要连同冻结 channel basis 与 target projection 一起切换，直接改 `output_rank` 不能把旧 256 维目标变成 16 维。

## 范围与原有结果

本次整理验证的是 CPU 前向/梯度等价、配置接线和手动训练保存流程；没有重跑 GPU 实验，也不提供新的 native 增益结论。原 Local/Global/R16 拟合失败及最后 external policy gate 的 NO-GO 结论保留，privileged-correction/OPD 研究队列仍停止。结构 preset 是读代码/手动改配置的起点，不代表推荐再次搜索这些超参数。

`adapter` 与 `pixel_views` preset 提供构造 API 和结构查看；**本工作区尚不包含任意新配置的 PTD native runner、标签 scorer 或 external teacher launcher**。原 native runner 是已锁的研究复现入口，其脚本/配置/数据 hash 必须一起遵守，不能直接改旧 `CONFIG.json` 再运行。需要连接新的 native 实验时，按 [CODE_MAP](CODE_MAP.md) 中实际两遍解码接口另建新运行记录。

CPU 检查：

```bash
.venv-ptd-audit/bin/python -B -m pytest -q tests/test_desta3d_workbench.py
```

## 双专家 native R16 方法（当前新实验）

完整实现与调参导航见 [DUAL_EXPERT_NATIVE.md](DUAL_EXPERT_NATIVE.md)。三因子网格为 K、rho、T/S 权重；本轮结果待真实 native 调参完成后填入。
