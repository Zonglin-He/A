# 新工作区运行 3EED Qwen 单帧指代

本机工作区位于 `/home/administ/3eed`，研究代码位于 `qwen-3eed`。当前论文主方法 `ray_equivalent_extent_2d_rank` 对两个已锚定的姿态分别做三轴观测等价投影，再以首次 Qwen 二维框排序。 [主方法登记](../../QWEN_CURRENT_METHOD.json) · [论文稿与结果](../../studies/3eed_extent_2d_factorial_20261128/MANUSCRIPT_ZH.md)

## 环境

| 项目 | 本机配置 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.5 LTS x86-64 |
| 硬件 | Core i5-14600KF，32 GB 内存，RTX 5090，32 GB 显存 |
| 驱动 | NVIDIA 580.178.04，GPU `sm_120` |
| Python | 3.12.16 |
| PyTorch | 本机现有 `2.15.0+cu132`，配套 `torchvision 0.29.2` |
| Transformers | `4.57.1`，与历史 Qwen 前端相同 |

新机沿用已验证 BF16 CUDA 运算的本机 PyTorch。历史生成回执使用 `torch 2.7.0+cu128`。CLI 在封存时记录实际 PyTorch、CUDA、Transformers、GPU 型号及显存峰值。推理使用一张 GPU、BF16、SDPA、贪心生成，以及同一批次交错提交两条描述的 3D 和 2D 请求。原 prompt、解析器、图像大小限制、生成长度和权重校验复用锁定代码。

```bash
cd /home/administ/3eed
/home/administ/miniconda3/envs/torch-env/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r qwen-3eed/tools/3eed_recovery/requirements.txt
```

几何阶段复用 3EED 的 Waymo 视角转换；此 CLI 的 Qwen 和区域几何路径可直接运行在已配置 PyTorch CUDA 的环境中。3EED 原生源模型的 PointNet++、RoBERTa 和源模型权重属于 `v1` 的另一条推理路径。

## 数据与权重

仓库外的 `assets/` 保存数据、模型及上游 3EED 源码。

```bash
cd /home/administ/3eed
export HF_HOME="$PWD/assets/hf-cache" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
model_revision=ebb281ec70b05090aa6165b016eac8ec08e71b17
dataset_revision=129fe3cc45722423c917943a4480ba454eb352bd

.venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --revision "$model_revision" \
  --exclude README.md .gitattributes \
  --local-dir "assets/models/Qwen3-VL-4B-Instruct/$model_revision"
.venv/bin/hf download RRRong/3EED 3eed_dataset.zip --repo-type dataset \
  --revision "$dataset_revision" --local-dir assets/archives

printf '%s  %s\n' \
  2ba7acd85b5240ec8acb6b9684a38deacafe873375fab4e3262183ef469fd809 \
  assets/archives/3eed_dataset.zip | sha256sum -c -
mkdir -p assets/data
unzip -n -q assets/archives/3eed_dataset.zip '3eed_dataset/*' -d assets/data
```

模型目录的每个文件均由前端按 [MODEL.json](../../studies/3eed_frozen_qwen_metric_grounding_20261120/MODEL.json) 核对。数据集和模型来源为 [3EED 发布页](https://huggingface.co/datasets/RRRong/3EED/tree/129fe3cc45722423c917943a4480ba454eb352bd) 和 [Qwen 模型快照](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17)。

上游转换源码在 `assets/upstream-3eed`，固定在 `843a901f5367a6be5cc484b97c6ecad462961dfa`。复建时可从 [worldbench/3EED](https://github.com/worldbench/3EED/tree/843a901f5367a6be5cc484b97c6ecad462961dfa) 获取该提交并检出。

## 推理和评分

```bash
cd /home/administ/3eed
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRITON_DEFAULT_FP_FUSION=1
tool=qwen-3eed/tools/3eed_recovery
dataset="$PWD/assets/data/3eed_dataset"
model="$PWD/assets/models/Qwen3-VL-4B-Instruct/ebb281ec70b05090aa6165b016eac8ec08e71b17"
upstream="$PWD/assets/upstream-3eed"
mkdir -p "$PWD/runs"
run_root=$(mktemp -d "$PWD/runs/ordinary-smoke.XXXXXX")
case_dir="$run_root/input"
run_dir="$run_root/output"

.venv/bin/python "$tool/prepare_ordinary.py" --dataset "$dataset" --output "$case_dir" --per-platform 4
.venv/bin/python "$tool/run_qwen_3eed.py" --upstream "$upstream" predict \
  --model-dir "$model" --manifest "$case_dir/manifest.json" --output "$run_dir"
.venv/bin/python "$tool/run_qwen_3eed.py" --upstream "$upstream" score \
  --dataset "$dataset" --manifest "$case_dir/manifest.json" --labels "$case_dir/labels.json" \
  --run "$run_dir" --output "$run_dir/metrics.json"
```

准备阶段通过公开账本中的匿名 UID 匹配 Ordinary Validation 原始帧槽位，按平台和路径排序，每个烟测帧取一条描述。保存到推理清单的内容只有图像、LiDAR、相机白名单字段、原始描述、相机视角、固定采样种子和 UID。评分命令先验证预测和清单的 SHA-256，再打开对应标注槽位。两个现有评分实现交叉检查直立有向 IoU，报告严格的 `>0.25`、`>0.50` 和 mIoU，汇总时平台等权。解析失败的预测计为零。

`predictions.json` 记录当前 `ray_equivalent_extent_2d_rank` 以及高度加二维、历史射线加高度、历史射线加三轴与冻结 `C_ray_simple` 四组对照，五组共用同一份 Qwen 回答。`seal.json` 以 `primary_arm` 标记本次主方法，并绑定输入、方法和预测。历史三臂、四臂封存可用各自的原 manifest 和标签独立评分。`--all` 准备公开账本中全部 9,310 条成员做一次新运行，按当前清单组批并重新生成回答。旧运行的私有批次次序与缓存由其原协议和封存记录定义。

带有固定正式登记 SHA 的历史研究脚本从其原提交复跑。三轴四臂实验的原提交为 `46c68c2cc15a79d5b738f8bddeb4a0f96ae0de18`。在本机可用同层独立工作树访问原运行目录和资源。 [高度加二维原登记](../../studies/3eed_qwen_extent_height_only_20261122/REGISTRATION_HEIGHT_2D.json)

```bash
git worktree add --detach ../qwen-height-2d-replay 46c68c2cc15a79d5b738f8bddeb4a0f96ae0de18
```

## 新机验收

主方法登记后，恢复 CLI 用 Waymo、Drone、Quad 各 4 条完成 GPU 推理和独立评分。三轴加二维烟测的三平台等权 mIoU 为 65.30%，Acc@50 为 83.33%。登记入口与既有 873 条三轴候选回放逐框一致。烟测用于检查接口和兼容性，论文报告使用封存面板。

2026-11-28 已验证 RTX 5090 的 BF16 SDPA、固定图像预处理与三平台四查询组批。公开数据 ZIP 的目录索引匹配全部 9,310 条 Ordinary Validation UID。使用发布包内 Waymo、Drone、Quad 各 4 帧，载入完成 SHA-256 校验的锁定权重，实际生成 24 份 Qwen 回答并独立评分，全部解析有效。模型校验、加载和推理合计 26.36 秒，PyTorch 显存分配峰值 9.43 GiB。

| 12 条烟测，三个目标平台等权 | Acc@25 | Acc@50 | mIoU |
| --- | ---: | ---: | ---: |
| 冻结 `C_ray_simple` | 75.00% | 66.67% | 50.97% |
| `ray_equivalent_height_only` | 75.00% | 66.67% | 56.09% |
| 三轴 `ray_equivalent_extent` | 75.00% | 66.67% | 57.57% |

该历史烟测的高度对照对 9 条描述收缩顶面。2,067 条 Selection 射线的命中状态均保持，首交深度最大差 `7.11e-15 m`。两个评分实现的 IoU 最大差为 `6.66e-16`。已验证被篡改的预测在读取标注前被封存校验拒绝。这批样本只用于验收新机的运行能力。

完整数据包通过 SHA-256 校验并解包到 `/home/administ/3eed/assets/data/3eed_dataset`。全部 9,310 条 Ordinary Validation 已准备到 `/home/administ/3eed/runs/ordinary-full-input/manifest.json`，其中 Waymo 3,782 条、Drone 2,792 条、Quad 2,736 条。完整数据与权重的传输日志在 `/home/administ/3eed/runs/download-progress.log`。

本机烟测结果在 `/home/administ/3eed/runs/ordinary-smoke-run`，可查看 `seal.json`、`metrics.json`、`geometry-check.json`。从完整解包目录独立重跑的结果在 `/home/administ/3eed/runs/full-install-smoke-run`。两次 `predictions.json` 逐字节一致，评分相同。
