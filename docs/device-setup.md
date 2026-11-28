# 换机说明

新设备先取 GitHub 上的代码和公开实验结果。只读报告时，直接打开 [仓库首页](../README.md) 即可。

## 取回仓库

```bash
git clone --branch main --single-branch --depth 1 https://github.com/Zonglin-He/A.git
cd A
git log -1 --oneline
```

按旧分支或早期提交找资料时，先从 [项目和分支索引](research-index.md) 打开相应版本。需要在本地查看 `main` 的更早提交时，运行 `git fetch --unshallow origin`。需要取其中一条研究分支时，运行 `git fetch origin <分支名>`，然后用 `git log FETCH_HEAD` 查看。

## 核对当前方法

S5-pass 代码在 `research/3eed-hypothesis-verification-20261128`。使用 [论文登记](../QWEN_PAPER_METHOD.json) 核对冻结 Qwen 修订、默认 `s5_pass` 入口和 [Verifier 权重哈希](../studies/3eed_minimal_verifier_20261128/S5_PASS_METHOD_LOCK.json)。可按 [3EED 推理和评分说明](../tools/3eed_recovery/README.md) 从本机资产恢复权重并复跑。

| 要运行的内容 | 准备方式 |
| --- | --- |
| 看实验结果、论文图、逐条公开分数 | 克隆后直接打开 [结果索引](research-index.md)。 |
| 用已准备的帧和已解析回答运行 Qwen 几何读出 | 使用 CUDA 环境和锁定的 Verifier 权重，按 [几何接口](qwen-grounding.md#推理入口) 传入 `frame` 与 `replies`。 |
| 重新运行冻结 Qwen 推理 | 准备 CUDA 环境和登记修订的 Qwen3-VL-4B-Instruct。按 [默认论文运行器](../tools/3eed_recovery/README.md#推理和评分) 准备输入。 |
| 复跑 WildDet3D 或 Talk2Car 的补充实验 | 将本地权重和数据路径填入 [外部复测说明](../studies/3eed_height_only_transfer_20261123/REPORT_ZH.md) 的变量。Talk2Car 调参另有 [固定协议](../studies/3eed_talk2car_parameter_tuning_20261123/PROTOCOL.json)。 |
| 复跑旧版 3EED 或 STVG | 使用 [旧版 3EED 运行说明](../v1/RUN.md) 或 [STVG 方法说明](method.md) 对应的环境与资产。仓库根目录的 `environment.yml` 是 STVG 的环境定义。 |

公开仓库保存实现、协议、结果表、论文图和哈希记录。复跑整套实验还需要从自有备份恢复相机图像、LiDAR、标定、已封存的输入和回答，以及相应模型权重。沿用所选实验报告里的目录和环境变量；部分旧封存回执记录了生成时的绝对路径。 [3EED 数据来源](https://github.com/worldbench/3EED) · [冻结 Qwen 权重修订](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17)
