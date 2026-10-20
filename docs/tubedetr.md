# TubeDETR：独立迁移与执行状态

本分支从 `c5afd0a7584e98f47fc8e0bde2d92b54b9ba488e` 出发。用户在 2026-10-20 授权 TubeDETR 的两个跨域方向及完整 Source／Ours 配对评价；TA-STVG、CGSTVG 和外部 TTA baseline 不在本机重复执行。旧五行结果及原父视频平均口径保持不变。

## 方法与接口

TubeDETR 官方源码完整保存在 `external/TubeDETR/`，版本为 `d70c0eed94da09a9a5e4329c937230ecfb1d3ee6`。上游源码没有修改。入口仍为 `python -m stvg`，增加 `--backbone tubedetr`。

Source 使用 TubeDETR 自己发布的 MDETR-init VidSTG／HC-STVG2.0、224 分辨率、k=2 checkpoint，优先恢复 `model_ema`，要求覆盖所有学习参数。发布文件中的 `pool_size=2` 显式映射为当前源码的 `stride=2`。只允许再生确定性的时间编码和忽略旧 Transformers 的确定性索引 buffer；不接受缺失学习参数。构造时不再下载 ResNet／RoBERTa 初始化权重，随后必须完整恢复 checkpoint 中的对应权重，分词器仍使用原 RoBERTa。

两个 checkpoint 的预期指纹来自本仓库历史已保存的官方资产回执，下载后必须重新计算验证：

| Source 训练域 | 官方 Drive ID | 预期 SHA256 |
|---|---|---|
| VidSTG | `1FaKT54s7LWcTgW562OXpVWHPXxNG7INu` | `2802c66049b2e7986a826b4bbd291eb8521446cfba6fee03b56729438bf44213` |
| HC2 | `1mkY1vv1U_pSVRj4JRY6uYTABKONQEh_w` | `c646a8b7131bb28806c350e16479cd8bb31d4425ccb15129b26a6e4d38dd4222` |

适配器捕获真实 space-time decoder 的 content 输入 `[T,1,256]` 及冻结 memory。8×256 临时 content 只加在八个观测位置对应的全网格节点；每条 query 新建状态和 Adam，结束和异常时由原 `fit` 清零。编码器输出在本条 query 的所有 OPD 步内复用。模型权重及其梯度保持冻结。

初始化、鲁棒共识、双尺度32候选、反向配对噪声、混合密度 OPD、更新步数、温度、学习率、gamma、alpha 和种子直接复用 [固定配置](../configs/default.json)。原 calibration 数值也直接迁移；其中 TA-STVG 校准来源和模型哈希不能解释成 TubeDETR 专门校准。接口与 checkpoint 配置在 [适配配置](../configs/tubedetr.json)。本轮没有新增参数搜索。

一个 ReferDINO 在同一组八帧上产生一条轨迹，同时用于初始化和 OPD。常驻专家运行在隔离进程中，每次只接收 RGB 和文本；每次调用前固定原专家随机种子。专家在查询间驻留 CPU，避免持续占用 GPU；不另加载独立 GroundingDINO。冷启动、传输和每条 query 的计算成本分别记录。

TubeDETR 原生只有一个完整时间网格，内部 k=2 负责 slow/fast 编码，不复制 TA-STVG 的双 offset 调用。Source 时间区间使用官方 `PostProcessSTVG` 的原生 MAP 和 end+1 规则。Strong 使用同一 Source 在448分辨率的后验占0.5、原生六层时间后验平均占0.5，然后调用原物理帧 MBR。OPD 不修改最终 Strong 区间。

## 数据版本与跨机器一致性

正式 HC 清单的关键问题是文件版本，不是样本计数的别名：

| 资产 | 已核对的数量 | 身份与边界 |
|---|---:|---|
| 历史正式 HC `val_v2.json` | 3482 clips/query、237父来源 | 历史 SHA256 `2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156`；必须取得并核对原文件，不能由同名清单推定 |
| TA-STVG 作者 `data.tar` 中 `hc-stvg2/annos/test.json` | 1902、227父来源 | 当前取得的文件 SHA256 `61f7a2a74d39c13dd380540d4ac819edb7abcde1826d44229e539d128bf3260c`；不是3482清单 |
| `ShijianW01/hc-stvg2` 的 `annos/test.json` | 1901、227父来源 | SHA256 `543ef809f31ed60f1b8a0514ea185ad4323596d63050087bfa47f2557df8026c`；名字集合是前述1902的严格子集，少1条 |
| VidSTG 官方 `test_annotations.json` | 4610条原始关系记录，展开10303句，732父视频 | SHA256 `ef769b9823ce2a4da8ff190ab69ccc77b8965181c58384fadc1f81e96fc4f2a7`，与历史完整评价集一致 |

3482的单位是不同视频片段／query；237是去除片段序号与扩展名后的父来源聚类数，不能把237当作评价样本数。历史3482文件的字节指纹已追溯，但原始文件的本机重获和官方发布包链条尚未完成。本机不以1901／1902替代正式HC评价。

VidOR作者发布的验证视频包 SHA256 为 `40dd102b43ecd2ed70f2dc38ee08258315e25919d2f1fb8f43bb235740a83a25`，验证标注包为 `65a06594c7735d8a0c384144ec2acb2227f715ccac2d03a6dee4a507de201516`。标注包已经完整核验，与历史回执一致；媒体完成状态以本地下载回执为准。VidSTG的测试视频来自VidOR validation，不能错误下载VidOR未标注的test替代。

`tools/build_manifests.py` 从原标注建立清单和独立标签，要求完整3482/237或10303/732；缺视频、目标轨迹缺失或数量不符都会失败，不过滤失败 query。输入延续现有物理支持：HC按原默认FFmpeg输出、6.4fps固定节点；Vid按零基 used_segment、至多128节点。标签端点保持已有数据集语义，不用评分结果修正。

这里的原生输出检查限定为：TubeDETR在上述预先固定输入网格上的完整网络输出，与注入零content的decoder重放一致。TubeDETR论文默认的5fps／最多200帧数据加载路径与这个统一项目网格不同，不能把当前接口检查称为论文默认协议的准确率复现。正式采样协议仍须在跨机器清单中明确锁定；不能在看到目标成绩后改采样。

跨机器应比较同一原标注哈希、完整查询集合、媒体字节、文本、帧网格、标签和指标实现；本地路径不进入 `population_hash`。仅交换3482／10303这样的数量不足以完成统一。目前尚无第一台机器对新规范化清单哈希的回执，不能宣称已经双机一致。

原HC128、Vid128已曝光的事实保持披露；新选的两条工程输入也登记为开发曝光，不据其GT结果调参。历史逐query曝光映射没有随Git提供，不能把完整划分称为未曝光测试集。

## 环境与解码边界

本机独立环境是 `.conda/tubedetr` 指向的 Python3.10 venv，复用本机 PyTorch2.7.0+cu128／torchvision0.22.0+cu128；不修改其他项目环境。Python版本与原机器3.11不同。ReferDINO的已有兼容CUDA源码已在架构12.0编译。Transformer版本为4.52.4。

本机已从官方 FFmpeg8.0.1 源码构建只供解码的二进制，指纹与历史原二进制不同。路径和哈希仅在本地 `runtime/tubedetr/assets.json` 登记，并绑定到资格检查和每次运行；历史 `stvg/decode.py` 的原指纹没有改写。新适配器调用相同两条数据集解码函数。**尚未证明与第一台机器原二进制逐像素一致**，不能把同一版本号当作该证明；正式对齐须核对工程输入的实际RGB哈希和物理网格。Source/Ours配对本身始终使用同一批解码数组。

## 工程资格检查与正式运行

结构测试只验证接口；完整资格检查必须用完整源权重和真实输入，检查：原生输出、零状态恢复、实际非零有限梯度、完整固定更新、适应后的重新原生编码、全部网络状态哈希不变、同八帧证据和物理网格。未产生实际pass回执前，不登记为模型已跑通。

```bash
.conda/tubedetr/bin/python -B tools/check.py
.conda/tubedetr/bin/python -B tools/prepare_engineering_inputs.py
.conda/tubedetr/bin/python -B tools/setup.py --prepare --tokenizer runtime/tokenizers/bert-base-uncased
.conda/tubedetr/bin/python -B tools/qualify_tubedetr.py \
  --hc2-request data/evaluation/engineering_hc2.json \
  --vidstg-request data/evaluation/engineering_vidstg.json

# 仅在完整清单、统一协议及真实资格检查都已经存在后运行。
.conda/tubedetr/bin/python -B -m stvg evaluate --backbone tubedetr \
  --manifest data/evaluation/hc2.jsonl --output outputs/tubedetr/vid_to_hc2 \
  --protocol data/evaluation/protocol.json --qualification results/tubedetr/qualification.json
.conda/tubedetr/bin/python -B -m stvg evaluate --backbone tubedetr \
  --manifest data/evaluation/vidstg.jsonl --output outputs/tubedetr/hc2_to_vid \
  --protocol data/evaluation/protocol.json --qualification results/tubedetr/qualification.json
```

协议JSON需有 `status=frozen` 和 `populations` 下对应数据集的 `population_hash`。运行还绑定代码、配置、checkpoint、解码资产和资格回执；逐query完成后原子写入不可覆盖的预测与哈希回执，断点恢复必须验证这些身份。全部完成后才生成 `sealed.json`。推理进程不接受标签路径。

## 新评分口径

```bash
.conda/tubedetr/bin/python -B -m stvg score --aggregation query --arms Source,Full \
  --manifest data/evaluation/hc2.jsonl --labels data/evaluation/hc2_labels.jsonl \
  --predictions outputs/tubedetr/vid_to_hc2 --output outputs/tubedetr/vid_to_hc2/metrics.json
```

主表按query平均 `m_vIoU、vIoU@0.3、vIoU@0.5`，严格使用官方 `vIoU > threshold`。另存等权父来源平均及父来源聚类bootstrap；query estimand 的聚类抽样每次以抽中的query总和／query总数计算，不能误算成父均值。

同一评分器支持明确的五行 `Source,Spatial,Temporal,WithoutOPD,Full`，供第一台TA-STVG使用，不能据此称本机完成了TA-STVG消融。按用户表格明确列名实现5项主定位指标和4项补充指标；另存未观测实际节点sIoU和分母。时间边界MAE按每条query的物理FPS换算成秒；没有GT观测节点时记null并报告分母，不能假装有观测证据。旧 `--aggregation legacy-parent` 及旧JSON不改变。

效率保存解码、Source、Strong、ReferDINO、适应和端到端耗时，保留cold-load成本；峰值按父进程分阶段allocated memory和串行专家进程的峰值合并，说明它不是NVML整卡reserved memory。最终表必须用完整集合的平均，不能把工程样例或缓存命中成本当全量平均。定性结果只能在预测封存、独立评分后，按预定类别选取空间改善、时间改善、共同改善和失败例，同视频同帧并排Source／Ours／GT。

本文件描述实现和可执行流程，**不代表已完成四个TubeDETR正式评价组合，也不包含新准确率结论**。

已有单专家真实GPU重复性回执见 [teacher_service_check.json](../results/tubedetr/teacher_service_check.json)：完整checkpoint哈希匹配、两个相同八帧调用逐位一致、一个专家、零独立检测器。这不代替完整学生流水线资格检查。CPU与物理帧工程证据见 [engineering_checks.json](../results/tubedetr/engineering_checks.json)，当前范围见 [STATUS.json](../results/tubedetr/STATUS.json)。

补充工具：`tools/acquire_tubedetr_assets.py` 对已授权的四个下载做有限断点恢复；`tools/prepare_tubedetr_run.py --wait-for-assets` 在资产完整后接续真实模型资格检查与完整Vid清单生成，在缺少统一协议时明确停住。`tools/report_tubedetr_efficiency.py` 只汇总完整封存总体；`tools/render_tubedetr_qualitative.py` 只在独立评分后绘制同帧Source／Ours／GT，并披露确定性事后选例规则及不存在的类别。
