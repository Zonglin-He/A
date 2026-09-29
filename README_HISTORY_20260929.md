# 最新：源可控性与同范数位置对照已完成

已按附件完成自由merger→冻结输出列空间→旧mask链，以及同16源的early/late×correct/wrong无更新对照。自由/列空间在两个监督源上改善，但等效latent更新巨大，不能据此声称易学。位置对照未支持“提前mask即可修好”：正确性差值early−late，时间−4.899pp CI跨0、空间+.1298pp CI跨0；错误负控本身退化与全部负尾保留。

先看[位置主表/匿名16例/全部CI](results/desta3d_v3/2026-09-28/SOURCE_LOCATION.json)、[裁决与局限](results/desta3d_v3/2026-09-28/SOURCE_LOCATION.md)、[三层正控表](results/desta3d_v3/2026-09-28/ACTUATION_CHAIN.md)。location144输出/0optimizer，6CPU控制、432几何、7628margin、174+36汇总二核通过。当前所有GPU结束，累计42173.00594093104秒cap=null；源GT诊断非TTA，无target/OPD/64。后续context机制需先固定operator/幅度另登记，当前未跑。外部pixel baseline仍在授权内、须先official smoke。

## 以下为历史记录，当前状态以本节为准

# 最新：三层源可控性诊断已完成并独立核验

时间病例的自由残差与冻结输出列空间控制均把tIoU从29.17%提高至68.06%；空间先暴露坐标类CE与原生全词表argmax的支持差别，保留格式失败后，仅修loss分母得到sIoU59.65→74.88%，列空间控制为73.83%。原模型/门控全部冻结，固定30步、无best选择。

关键限制：列空间控制等效latent改变量为原latent的2698/5310倍；它证明两个源病例存在可达方向，不证明reader容易学会，更不是OPD/target收益。旧mask与监督控制不是纯位置单因素。下一检验同norm的before/after reader和正确/错误mask；尚未运行。

见[完整对照表](results/desta3d_v3/2026-09-28/ACTUATION_CHAIN.md)、[机器聚合及校验](results/desta3d_v3/2026-09-28/ACTUATION_CONTROL.json)、[解释与实现](docs/desta3d_v3/ORACLE_FAILURE_DIAGNOSIS.md)。所有失败、旧oracle负数和原始数据本地保留；公开不含权重、标签、caption、预测raw。GPU累计41976.21386800704秒，cap=null。官方外部权重下载已完成，后续pixel baseline尚待独立smoke/qualification。

## 以下为历史记录，当前状态以本节为准

> **Current 2026-09-28:** source native free-actuation control completed: temporal success, spatial coordinate-CE/native-grammar mismatch. Matched full-vocabulary spatial control is running. See [current review](REVIEW_START_HERE.md). All older notes below are historical.

> **Current 2026-09-28:** source GT branch-latent oracle completed with matched wrong controls; correct-evidence advantage not established. External downloads resumed only for the pixel baseline. No OPD result. See [latest review entry](REVIEW_START_HERE.md) and [oracle table](results/desta3d_v3/2026-09-28/LATENT_ORACLE.md).

> 2026-09-28 更新：全量源训练已按用户要求停止，新外部证据路线请从 [REVIEW_START_HERE.md](REVIEW_START_HERE.md) 和 [结构图](docs/desta3d_v3/EXTERNAL_PRIVILEGED_OPD.md) 开始。代码/CPU接口已实现，teacher权重下载中，OPD效用未测。

# A — STVG adaptation research code

**2026-09-28 最新审阅包已更新。** 从 [REVIEW_START_HERE.md](REVIEW_START_HERE.md) 开始；包含当前 DESTA-3D v2 的代码、结构、单因素对照、失败更正和未解问题。

历史实验和独立审计保留。当前正在准备首次外部teacher及四视图PTD资格验证；五项审阅修复和16项CPU控制已完成，官方权重下载中，尚无teacher GPU资格结果。用户授权Luna max每30分钟检查并由主代理处理异常。尚未建立可靠目标TTA净收益，也未晋升生产方法。

| 阅读内容 | 入口 |
|---|---|
| 推荐阅读顺序与具体审查问题 | [REVIEW_START_HERE](REVIEW_START_HERE.md) |
| 计算图、训练和实际更新/冻结路径 | [结构说明](docs/DESTA3D_V2_ARCHITECTURE.md) |
| 完整正负汇总与解释范围 | [2026-09-28结果](results/desta3d_v2/2026-09-28/README.md) |
| 优化器恢复事故及其他版本更正 | [CORRECTIONS](docs/DESTA3D_V2_CORRECTIONS.md) |
| 可复制给其他GPT的提示词 | [EXTERNAL_REVIEW_PROMPT](docs/EXTERNAL_REVIEW_PROMPT.md) |
| CPU检查和公开范围 | [REVIEW_VALIDATION](docs/REVIEW_VALIDATION.md) |
| 外部依赖与固定版本 | [DEPENDENCIES](docs/DEPENDENCIES.md) |
| 源码到公开副本的hash映射 | [PUBLICATION_UPDATE_20260928](docs/PUBLICATION_UPDATE_20260928.json) |
| 既有DeCoTA精简模块 | [decota_final_simplified_v1](methods/decota_final_simplified_v1/) |

此仓库不包含视频、数据集、标注、权重、私有manifest/授权附件、逐样本预测或raw梯度/logits。原科研证据在本地保留；完整GPU runner依赖这些原件及严格pins，仅克隆不能重跑全实验，不要禁用其完整性检查。

## CPU模块使用

```bash
python -m pip install -r requirements-core.txt
python examples/desta3d_v2_cpu.py
python -m pytest -q tests/test_desta3d_v2.py tests/test_desta3d_v2_training.py tests/test_desta3d_v2_tta_objective.py
```

CPU示例使用合成特征，不是视频定位demo或任务精度结果。更完整的本次CPU测试见验证记录。

历史材料继续保留：[2026-09-27结果](results/desta3d_v2/2026-09-27/README.md)、[旧README快照](docs/README_SNAPSHOT_20260927.md)。旧“源训练running、target TTA未开始”和连续optimizer说明已过时；以上最新入口与更正优先。递归依赖里有历史和失败runner，它们不是当前方法或自动续跑指令。
