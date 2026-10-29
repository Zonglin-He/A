# GREAT / PIADv2：固定权重 TTA 小规模诊断

本包包含新增的无标签适应、数据准备、预测封存、独立评分、权重恢复和兼容脚本。上游固定为 `yawen-shao/GREAT_code@8ab28ec4dd7bbfe4d3a70aa9fe9b7ef1295d0cef`；不打包完整上游源码、权重、图像、CoT 原文、点坐标、标签或原始预测。

## 本轮结果与边界

12 个按公开文件路径哈希确定的不同点云文件，覆盖 5 个 affordance 类别。原序和去序精确几何哈希均为 12 个不同值；尚未核实它们是否来自独立物体实例，样本级区间仅作描述。Source AUC/aIOU 为 64.3496/9.1349，零更新三视图 ensemble 为 67.3853/10.2934（百分数）；但 aIOU 仅 4 例改善、7 例退化、1 例不变，均值受少量正尾影响，不能称稳定收益。三步一致性更新为 64.3193/9.1282，三步熵更新为 64.3498/9.1356，均未产生实质一致改进。

随后只改变 student 更新视图的 canonical-v2，把原序预测对齐到同一冻结 ensemble teacher，保持教师、LayerNorm、学习率、步数、随机种子及每步 forward 数一致。其三步结果为 64.3585/9.1345：teacher MSE 下降，但任务变化依然很小。该 v2 使用已经曝光 v1 标签结果的开发集合，不构成独立测试。结论为**不采纳当前参数 TTA**，不据此否定整个研究路线。完整聚合指标和负尾见 `results/pilot_summary.json`。

最大 allocated CUDA memory 为约 1.15 GB，仅覆盖使用数据集离线 CoT 的评估模型，不含现场 InternVL 推理。诊断计时含重复一致性等审计开销，不能直接作为最小部署延迟。

## 安装与恢复上游

```bash
python3.11 -m venv env
. env/bin/activate
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
git clone https://github.com/yawen-shao/GREAT_code upstream
git -C upstream checkout 8ab28ec4dd7bbfe4d3a70aa9fe9b7ef1295d0cef
python scripts/prepare_source.py
python scripts/download_text_assets.py
```

`compat/GREAT_compat.patch` 给出最小兼容差异：替换文本路径占位符、用配置构造再严格加载完整模型权重、令文本张量跟随实际模块设备。PointNet++ 保持上游纯 PyTorch 实现。

官方当前源码写 `roberta-base`，但下载权重的词表为 40000；原始 git 历史使用 `pdelobelle/robbert-v2-dutch-base`，与该词表一致。本实现恢复历史 checkpoint 兼容配置，并记录文本资源固定版本/哈希；没有声称已消除论文文本编码器身份矛盾。

## 获取权重和数据

```bash
# 每个恢复成员验证官方 ZIP 的 size/CRC，记录成员 SHA256 和别名映射；不会执行 checkpoint 中的任意全局对象。
python scripts/acquire_model.py

# 完整官方数据 ZIP，支持断点续传；下载后仍应执行下面的全 ZIP CRC 检查。
python scripts/range_download_keepalive.py 'https://drive.usercontent.google.com/download?id=1rkWAZcIEfj5Rk7E3EOlh0gwDr4Hg14pp&export=download&confirm=t' downloads/Unseen_aff.zip --workers 16
python -m zipfile -t downloads/Unseen_aff.zip

# 先准备同一固定 pilot，只获取必要测试文件；没有按 GT 结果选样。
python scripts/remote_zip.py --url 'https://drive.usercontent.google.com/download?id=1rkWAZcIEfj5Rk7E3EOlh0gwDr4Hg14pp&export=download&confirm=t' --output data
python scripts/prepare_cohort.py --limit 12
```

权重恢复产物明确命名为 `model_only_crc_recovered.pt`，不是原始训练 archive。原始完整 checkpoint 路径和数据入口由上游 [README](https://github.com/yawen-shao/GREAT_code) 和作者 [issue #3](https://github.com/yawen-shao/GREAT_code/issues/3) 提供。相同 size+CRC 的重复成员可复用；这与完成原始整 archive 的 SHA256 验证分开。对关键结论，保留原始完整权重与恢复模型的逐张量等值审计。

## 推理、适应与独立评分

先运行工程契约检查，再封存实际预测，最后独立评分。更新器只接触图像、XYZ 和离线知识文本，标签由单独评分清单管理。

```bash
python scripts/check_tta_contract.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/run.py --manifest data/inference_manifest.json --checkpoint downloads/checkpoints/best_unseen_aff.model_only_crc_recovered.pt --output runs/pilot_v1 --device cuda --steps 0 1 3
python scripts/score.py --run runs/pilot_v1 --targets data/scoring_manifest.json

# 匹配的 canonical-student 机制诊断；保留 v1 输出目录。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts_v2/run.py --manifest data/inference_manifest.json --checkpoint downloads/checkpoints/best_unseen_aff.model_only_crc_recovered.pt --output runs/canonical_v2 --device cuda --steps 0 1 3 --objectives consistency
python scripts/score.py --run runs/canonical_v2 --targets data/scoring_manifest.json
```

LayerNorm 更新只涉及 `decoder.cmff.cross_atten1.layernorm` 的 1024 个 affine 参数。固定 Adam 学习率 `1e-4`，固定 0/1/3 步，不按 GT 选择步数、回滚或阈值；episodic 恢复参数、Python/NumPy/torch/CUDA RNG，保持 BN/dropout eval。评分前验证预测 SHA256 seal，报告各指标有效/未定义分母，并断言 source/ensemble 跨步数和目标一致。AUC、20 阈值 aIOU、sum-normalized SIM、MAE 遵循 native evaluator 语义。

## 机制判断边界

重排输入还会改变 FPS 的首个物理采样点，以及按输入索引截断的 ball-query 邻域；只控制 FPS 起点不足以把敏感性归因于 decoder。部分点云有明显序结构，因此任意排列是受控 nuisance 干预，不是自然域偏移。几何一致性也可能稳定错误功能区域。后续应先区分采样/邻域集合与固定索引融合的贡献，再做独立 held-out 验证。

公开 Unseen-aff/Unseen-obj 配置把 val/test 指向相同列表，训练脚本按 validation aIOU 选择 checkpoint；这可能导致 source 选模使用测试标签，但尚不能据此证明发布权重的具体训练过程。现有 [affordance TTA](https://lemonqc.github.io/WeaklyTestAffordance/) 也排除了“首次 affordance TTA”的新颖性说法。

所有本地原始计划、代码快照、原始预测、标签和媒体保留在私有工作目录；公开包只提供新增代码、最小补丁、协议摘要和可公开聚合结果。
