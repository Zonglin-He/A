# CG-STVG 迁移适配

本页记录第二个 backbone 的真实接口准备，不替换已登记的 TA-STVG 主入口，也不宣称已经完成 CG 的正式评估。数值配置仍取 `configs/default.json`；本项不重新选参、不训练模型、不新增专家。

## 官方来源与源域权重

官方论文：[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gu_Context-Guided_Spatio-Temporal_Video_Grounding_CVPR_2024_paper.html)。官方代码：[HengLan/CGSTVG](https://github.com/HengLan/CGSTVG)，固定提交 `f062d84b72aee8be4425a4a5ab20aa5b077ca142`，存于 `external/CGSTVG/`。没有改写 vendor 训练代码。

官方 README 指向 [Gstar666/CGSTVG](https://huggingface.co/Gstar666/CGSTVG)，固定 Hub revision `9b0d7b9680c23ae14071ff6ca89f63f6a0990b2d`。该仓库下载受访问权限限制，但本机已有两个文件；已与官方 Hub API 文件元数据所载完整文件 SHA256 核对，不需要重新下载。

| 目标域 | 源域检查点 | 文件 SHA256 |
|---|---|---|
| HC-STVG2 | `vidstg.pth` | `cbf3ee4036dab88de3314dc3a520ab0eb2b4d292a2796d12d0f3493c39eed9ef` |
| VidSTG | `hcstvg2.pth` | `020d77b1013e33d9ae58f01c5b60e77ba6eb3d05e04077a94fa8e4247d635a88` |

两文件均为 3,548,827,981 字节，以本地链接放在被忽略的 `checkpoints/cgstvg/`，原文件未移动。检查点具有 `model/model_ema/optimizer/iteration`，模型映射各 1608 项。适配器严格加载官方 `model_ema`，不使用检查点内 optimizer，不沿用其训练过程。源域文件名与官方哈希绑定，不能按目标域替换权重。

```bash
.conda/tubedetr/bin/python -B tools/setup_cgstvg.py --prepare --check --output outputs/maintenance/cgstvg_assets.json
# 独立进程，仅 CPU 构造与严格载入，不执行模型前向。
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B -m stvg.cgstvg --target hc2 --load-audit outputs/maintenance/cgstvg_hc2_load.json
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B -m stvg.cgstvg --target vidstg --load-audit outputs/maintenance/cgstvg_vidstg_load.json
```

本地 `runtime/cgstvg/model_zoo/` 只链接已有 RoBERTa 与 Video-Swin 初始化资产，ResNet 使用已有 torch 目录。以上准备不联网、不下载模型，也不新增检查点。

## 与 TA 接口的区别

CG 原生 `models.pipeline.CGSTVG.forward` 直接调用自己的 QueryDecoder，没有 TA 的属性分类器、`itq/isq` 两次重构。因此 `stvg/cgstvg.py` 单独捕获原生两个物理 offset 的空间 decoder 输入和时间 decoder 六层输出，不调用 TA 的 `SpatialReplay.reference_values`。

CG 的配置元数据将时间分支明确写为 `Strong CG-STVG only`。初始化使用的是对应源域已有 TA-STVG 校准常量的直接迁移，**没有重新在 CG 的源训练数据上校准**；`E_calibration.model_hash` 因而仍标识当时的 TA-STVG 权重。适配器在 `E_calibration_provenance` 中明确记录这点，所有常量和主配置数值保持不变。

CG 的空间 decoder 返回 `[refer_anchors, anchors_conf, fake_anchors]`，其中第一项形状为 `[层,1,T,4]`；必须先选择该项，再取最后层。临时状态仍为八个独立 256 维 content 状态，由既有物理位置／offset 映射注入；原双尺度 policy、几何反馈、初始化保留映射与简化参数不改。源域配置中的预测置信度和时间阈值按源模型原生 yaml 保留。

Strong 使用此 CG 模型本身的原生六层时间后验，以及同模型高分辨率最后层后验，保持原等权融合和物理区间 MBR；不借用 TA 的时间 logits。CG 的冻结视觉／运动／融合 prefix 与 TA 一样采用 fp16 autocast，实际 CG decoder 和输出头使用 float32，并要求缓存零状态 replay 完全一致。捕获复用已审计的 Swin 存储复用、瞬时 mask 和冻结模块 host staging；CG pipeline 仅增加显式精度边界与中间视觉张量的 host staging，没有替换其视觉计算。实际内存／吞吐需真机验证，不能沿用 TA 已测成本。

CG 和 TA 都使用顶层包名 `models/config/utils/datasets`，因此 CG 必须运行在新解释器，不能在已经载入 TA/ReferDINO 的进程内切换 `sys.path`。`official_imports()` 会拒绝已存在的异源同名包。ReferDINO 继续在既有独立教师子进程运行，同一批八框复用于初始化与 OPD。

## 推理标签隔离

上游 `PosDecoder.forward` 即便 `iteration_rate=-1`，也会在每层调用 `gt_info(targets)`，读取 `boxs/actioness`。预测分支仅以自己的置信度与时间预测选择上下文，这项 GT 读取在推理中没有被使用。

`inference_only()` 用实例级、作用域限定的源码替换，仅将该唯一死读取改为 `(None,None)`；它拒绝训练模式或非负 `iteration_rate`，退出后恢复原函数。没有给它输入真实 GT，也没有修改 vendor 文件或训练函数。捕获的 replay 输入去掉无用 targets 和原始 RGB 引用，只保留原生计算所需输入。

CPU 检查直接执行官方 `PosDecoder.forward` 的实际代码，用合成算子和合成标签核对原输出；再用禁止任何下标／迭代读取的 targets 检查补丁输出逐位一致。它证明的是死读取绕过与接口语义，不能替代真实模型权重下的推理等价验证。

## 已验证与尚未验证

已核对两个源权重的字节与官方哈希、CPU 安全检查点结构及实例级补丁。两个方向的完整官方 CG 模型均在独立、禁用 CUDA 的进程中完成 CPU 严格 EMA 加载，所有模型状态覆盖，仅忽略确定性的 RoBERTa `position_ids` 兼容项。构造过程中 RoBERTa pooler 的提示发生在最终 CG 检查点载入之前；随后严格模型状态覆盖，不把提示当成允许随机参数参与推理。

`stvg.cgstvg.Predictor` 已连接真实 Source 捕获、单 ReferDINO 初始化、相同双尺度 policy 和 CG decoder replay。两个已有真实输入的初版 GPU 工程检查均已完成：HC 执行40次更新、Vid执行30次，二者缓存零状态与实际捕获的 Source 框精确相同；收据为本地 `outputs/formal/checks/cg_hc2.json` 与 `cg_vidstg.json`。它们只验证两个工程样例，不等同完整面板评估或新增性能结论。

随后对官方 `generate_roi_feature` 增加纯同步优化：在原有 round／ceil／long 后一次性将整数框坐标复制为 CPU 列表；循环只用该列表解包坐标。原 GPU `bboxs`、特征 clone、切片、`torch.mean` 顺序和异常回退不变，不矢量化归约。补丁仅作用于真实官方方法，作用域结束恢复原方法，合成替代函数不会强行套用补丁。两个 dtype×两种模态的 CPU 测试均保持输出和 `encoded_feature` 梯度逐位相同。包括原核心检查，合计17项测试通过；此同步优化仍须对上述两个真实封存预测重放并核对全部数组、实测耗时，不能仅凭 CPU 测试宣布完成真机回归。

正式评估启动前，仍须确认最终代码的真机回归、物理位置及固定配置检查全部通过。此处没有 CG 的新增 vIoU 成绩，也没有把获得权重或两个工程样例描述成已完成迁移实验。CPU 严格加载收据保存在 `outputs/maintenance/`，GPU 工程收据在 `outputs/formal/checks/`；以实际收据为准，不把未产生的收据当作完成。
