# LLaVA-ST 主干：原双域128开发与锁定后全量

2026-10-22 用户明确将最初的“28来源”纠正为**原固定128来源**，并指定 LLaVA-ST-Qwen2-7B **作为 backbone**。本轮是独立的新主干运行，保留全部旧 TubeDETR/TA-STVG 实现、配置、封存预测和正负结论。不会恢复旧 OPD 或轻量传播头实验。

## 来源与可支持的比较

- [官方代码](https://github.com/appletea233/LLaVA-ST)：固定 `bacf6d61e1de27a78fc0025083aa445d3d41b0b3`。
- [官方合并权重](https://huggingface.co/appletea2333/LLaVA-ST-Qwen2-7B)：固定 `2f7261b55b0368e9c2971120b62c77545b3a792d`，14个文件共16,203,228,377字节。大文件按发布者 LFS SHA256 核验，小文件按 Git blob SHA1 核验，并记录全部本地 SHA256。下载比例、结构检查和真实GPU资格分别报告。
- [论文](https://arxiv.org/html/2501.08282v2)的 ST-Align 数据说明包含由 VidSTG 构造的指令数据。这里只有一个共享合并 checkpoint，不是两份分别只在源域训练的权重；不能把两目标域评价命名为已验证的严格源域到目标域迁移，也不能声称 VidSTG 未见。

VidSTG仍为10303 query/732视频，HC2仍为1902 query/227父来源；HC注释身份仍为 `61f7a2a74d39c13dd380540d4ac819edb7abcde1826d44229e539d128bf3260c`。每域开发面板沿用原128 query/128独立父来源，输入SHA分别为 Vid `c7df9b65aa5603b14a132615cc19f71a51a9bd59baf36f38bd28426334ff0ec4`、HC `4cf05614ed1fe083b0c117ad0b8e8ddd952b688ccbccb9a56fcdace5942b1664`。这些开发面板和旧主干全量指标已经曝光，新主干结果仍不能冒充未曝光的独立确认。

冻结协议见 [plan.json](../results/llava_st_development/plan.json)，来源边界见 [source_provenance.json](../results/llava_st_development/source_provenance.json)。计划在任何本轮真实GPU资格和开发指标前生成。

## 主干接口

使用官方模型类、原始参数和官方 prompt 预处理函数，不调用包含标注解析和出错后换下一query逻辑的官方数据集加载器。

1. 按现有数据集专用解码获得相同规范视频支持，再等距取样/必要时重复为官方100帧输入。时间token索引映射到实际选中的物理帧；重复物理帧只在内部特征上求均值，评价网格保持真实唯一帧，不伪造新时间节点。
2. 固定原文本加官方STVG任务后缀。原生生成使用官方温度0.01、单beam、最多1024新token；每query以seed41、文本和原物理网格的哈希重置随机种子。没有时间参数搜索或根据GT重试生成。
3. 原生100格时间token映射到实际物理帧，区间右端加1成为半开区间。**Source与Ours共用同一原生区间；没有Global6、MBR或GT裁剪**。自回归时间输出不伪装成旧模型的逐帧时间后验。
4. 原生输出为稀疏逐时框；在实际物理网格上做xyxy线性插值，生成框范围以外保持最近端点，Source/Ours完全相同。这是明确声明的本地稠密STVG评价适配，不是官方稀疏ST-Align成绩复现。
5. 空间特征来自真实第一级空间STP输出、时间压缩之前的 `100×81×3584`，保留原生LAPE。没有伪造256维特征、PCA或训练投影。旧证据/几何运算仅把四处固定256形状改为实际维度；256维含未观察节点的全部张量与最终DP数值精确回归。
6. 每query冻结整个主干，无训练或OPD。空间方法仍为同一个 ReferDINO 的八处观察、完整教师框、原共识初始化、BoxFixed/WLS、固定评分与精确DP。原冻结初始化标度直接迁移，明确不是新主干的源域校准。

缺少时间或框的生成结果保留在总体中，不能换query、读取GT补框或反复采样至成功。无有效轨迹时不进行空间修正，s/v按0评分；若只有轨迹缺失而时间有效，仍保留该原生tIoU。缺少时间则tIoU为0。所有数量和分母公开。

## 工程环境与资格门槛

隔离环境为项目外 `envs/llava-st`，读取旧环境依赖而只在新环境覆盖官方指定 Transformers 提交 `1c39974a4c4036fd641bc1191cc32799f85715a4`（4.40.0.dev0）及tokenizers0.15.2等推理依赖。旧TubeDETR/ReferDINO环境不改。复现入口为 `tools/llava_st/setup_environment.py`。

合并权重包含完整SigLIP视觉参数，构造相同视觉结构后全部从该checkpoint恢复，不重复下载/加载辅助基础视觉权重。旧config嵌套的词表字段显式展平为151660基础行加300坐标行；没有resize或随机重置。CPU meta结构的839个state键与发布index一致，总参数8,095,709,216；随机小模型只检查API，不是实际模型推理成功或性能证据。

所有文件完整校验后，先在两个域各一条预先指定的原开发首query上验证：真实生成有效、真实81×3584空间网格、八个不同观察、原教师证据、完整默认空间解码有限、权重前后哈希一致/无梯度。最多2次原生生成和2次八帧教师请求；不读标注，不换资格query。工程失败停止接续并记录failure，不自动扩大预算。ReferDINO仍在原环境独立进程执行，保存证据后释放临时输出及CUDA缓存，避免两环境依赖冲突。

## 有限选参与全量门槛

每域所有profile共享同一份完整128 GPU捕获，没有新网络前向；逐profile先封存完整128预测及receipt，再由独立CPU进程读GT评分。

| 顺序 | 参数 | 冻结取值 |
| --- | --- | --- |
| 1 | gamma初始化强度 | 0/.0125/.025/.05/.1/.25/.5/.75/1 |
| 2 | admission准入 | .05/.1/.2/.4 |
| 3 | guard保护 | .025/.05/.1/.2/.4 |
| 4 | link连接 | 0/.125/.25/.5 |
| 5 | gamma邻居复查 | 最多2个已登记相邻取值 |

原默认一直保存为参照，每域最多24个不同profile。每阶段选完整128等query Ours m_vIoU最大配置，严格并列保留当前配置；公开全部正负配置、dense/观察/未观察空间、无效输出和尾损，不做每query GT门控。此轮是空间数值选择，不根据全量重新调参。

两个域都生成 `selected_profile.json` 和总 `selected_profiles.json` 后，才依次执行Vid10303、HC1902。选择出的开发128预测逐字节绑定复用，其余query真实运行新主干；旧TubeDETR输出不混入。全量完整封存后另CPU评分，报告配对父来源bootstrap、生成失败数和统计分母。没有部分全量评分表。

## 执行与证据位置

```bash
.conda/tubedetr/bin/python -B tools/llava_st/download_assets.py
/home/wwww/stvg/envs/llava-st/bin/python -B tools/llava_st/prepare.py
/home/wwww/stvg/envs/llava-st/bin/python -B -u tools/llava_st/development.py --stage run
```

`prepare`只允许首次冻结计划。控制器可先启动等待完整权重seal，通过资格后继续，不因等待、低GPU或CPU评分判定停滞。所有GPU阶段共用 `runtime/gpu.lock`；推理在校验输入/代码前安装标注访问guard，标签hash和解析仅在完整预测seal之后的独立评分进程执行。

- 下载：`runtime/llava_st/download_status.json`、`checkpoint_seal.json`、`download_failure.json`（仅失败时）。
- 控制器/worker身份：`runtime/tubedetr_active_jobs.json` 的 `llava_st_controller`、`llava_st_worker`，并核对实际 `/proc`。
- 状态：`outputs/llava_st_development/status.json`、`worker_progress.json`、`failure.json`（仅失败时）。
- 日志：`logs/complete_llava_st_development.log` 和 `logs/llava_st_<domain>_<stage>.log`。
- 私有输入/捕获/教师缓存/原生生成文本/预测/receipt：`outputs/llava_st_development`，不上传。
- 公开资格、全部128指标/决策/选择锁、完整指标/报告和最终seal：`results/llava_st_development`。

代码、计划和CPU工程证据发布不能当作权重完整、真实GPU资格、128选择或最终全量结果发布。最终结果需要晚于终态的新发布回执；immutable seal里的publication_pending保持封存时含义。
