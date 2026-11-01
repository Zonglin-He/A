# R42：固定 native-point 接口的 3B→7B 容量诊断

## 裁决与总体结果

**不采用本轮 7B point 配置，不接 TTA。** 主指标 mIoU 的两平台平均为 22.645214%，同接口 3B 为 22.740349%，差 −0.095135pp；Quad +0.650931pp，Drone −0.841201pp。旧 G 为 24.727373%，7B 低 2.082159pp。这个结论只针对固定提示、原图与 point/映射管线，不表示 7B 一般无效，也没有比较 7B 的其他输出接口。

| 方法 | Drone mIoU% | Quad mIoU% | 等权平均% |
|---|---:|---:|---:|
| source | 12.742985 | 26.584141 | 19.663563 |
| R29_direct | 13.650161 | 27.004793 | 20.327477 |
| legacy_G | 15.978627 | 33.476120 | 24.727373 |
| 3B_point_compatible_fullmap | 15.414427 | 30.066271 | 22.740349 |
| 7B_point_compatible_fullmap | 14.573226 | 30.717201 | 22.645214 |
| 3B_point_compatible_oldsubset | 13.913294 | 29.610553 | 21.761923 |
| 7B_point_compatible_oldsubset | 13.705079 | 29.341481 | 21.523280 |
| 3B_som_compatible_fullmap | 9.559451 | 27.288389 | 18.423920 |

其他指标并非全部下降，必须同时保留：Quad 的 Acc25/50 和 Drone 的 Acc50 提高；按预锁 mIoU 主指标不能据此切换采纳标准，也不能将本轮称为所有指标均退化。

| 接口 | Drone Acc25/50% | Quad Acc25/50% |
|---|---:|---:|
| source | 26.923077 / 9.615385 | 47.115385 / 30.769231 |
| 3B_point_compatible_fullmap | 30.769231 / 7.692308 | 52.884615 / 32.692308 |
| 7B_point_compatible_fullmap | 30.769231 / 9.615385 | 54.807692 / 34.615385 |

7B 相对 source 有 19 条 IoU 改善、3 条恶化；相对同接口 3B 有 24 条改善、31 条恶化，其余相同。全图中只有 25 条 source query 改变，3B 为 89 条。source 回退是固定方法的一部分，所有回退均计入总体；没有只报告接受子集。1664 个 IoU 与 1045 条有序负比较全部保留，重复对照及负比较不是独立数据。

## 弃权、映射与身份边界

7B 的 208 条原回答中，100 条显式 null、9 条解析失败；99 条给出结构合法点，其中 41 条越过既定图像像素边界。最终 fullmap 49 条映射（Drone20、Quad29），另有无 mask1、重叠3、空 3D 组5。完整映射覆盖为 49/208，原兼容 3B 为 160/208（Drone79、Quad81）。mapped 不等于主体身份正确，null 也没有改成错误的正伪标签。

同一预封 24 条身份记录中，18 明确、6 歧义。7B 的 18 明确中，5 条与原注释一致、1 条矛盾、12 条没有唯一 mask；原 3B 为 8/8/2。不能只取已回答的 5/6 与 3B 的 8/16 相比，然后宣布总体监督更可靠：回答覆盖明显不同，而且这些是非专家、存在历史暴露的暂定身份假设，不是新 gold。6 条歧义不判对错，身份从未被最终 IoU 改写。

7B 的 5 条一致身份在 fullmap 下均可映射，旧 subset 只有 2 条可映射；完整映射也让 1 条错误身份映射成功。它解决覆盖，不保证语义。原注释 SHA 保持 6e382dc7cd7cdc4257a8d4811c1b640ca0d466eaab6b2b72fb1dde916c099097，没有重新看图重标或按小子集救分。

## 固定协议

对象为已暴露A208开发集，Drone/Quad各104条。原3B使用R37原始回答及R38兼容解析缓存；7B为官方Qwen2.5-VL-7B-Instruct固定revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`，实际参数8,292,166,656，BF16、SDPA、strict local safetensors，missing/unexpected/mismatched/error keys全空。

输入固定为逐字完整文字与一张未标记原图，不加裁剪或手工分数。每图256–1024视觉token，输入上限8192且不截断；greedy、beam1、96输出token、repetition_penalty1.05，seed0、TF32关闭。3B/7B功能性processor/tokenizer/chat-template文件相同；两者generation config文件仅版本元信息不同，有效解码参数一致。

每次7B输入核对历史3B的RGB数组、caption、序列化prompt、实际grid与token数量；另用原3B processor做CPU重构，与7B全部输入tensor逐项exact比较。历史3B未保存完整token向量，报告不冒称存在该缓存。CPU重构没有加载第二个生成模型。

最终解析使用原R38唯一point兼容合同（对象或singleton对象列表、原别名、可选惰性label），不宽松提取多位置、坏JSON或执行文字。坐标为实际grid×14模型输入像素，逐轴回原图，仅查唯一mask；null、越界、无mask、重叠或空3D组均保留。完整/旧subset组及原source最高分代表不变，输出仍是既有三维框。source、旧G、R29直接重排、同接口3B以及SoM均使用固定缓存；SoM为双图，资源不同需明确披露。

原8个true-square fixture单列为坐标能力诊断；208真实调用包括两条复用技术预检，总调用216。预检采用root在生成前明确的A208 canonical顺序过滤原12后首个平台，即A_drone_044/V_quad_002，不是按当前收益挑例。两条均通过软件与输入合同、结构可解析，合法3D映射为0仍保留；unknown不是软件失败，不能据此改提示或删样本。实际生成没有重答、best-of或参数更新。

原始token IDs先持久化，decode成功后立即atomic保存raw字符串，然后才解析或生成派生hash。216份raw、绑定、位置/unknown原因、208份八臂query/原框与资源回执全部封存后，才进入独立CPU审计与评分。预测seal：`23aabcf01c3c67cf19a146b136d96d69c092478b655659804127647113bf8385`。

## 资源与下载历史

| 测量 | 实际值 |
|---|---:|
| 权重和旧输入seal校验（模型计时前另列） |18.63秒|
| 模型加载 |12.26秒|
| 模型阶段墙时：加载、8合成、208真实及输入核对 |124.58秒|
| CUDA峰值allocated |17.163GB，约15.98GiB|
| CUDA峰值reserved |19.046GB，约17.74GiB|
| 进程host VmHWM |约4.87GiB|

峰值allocated/reserved在加载后重置统计，最终峰值仍包含常驻权重；加载峰值另有独立回执。这里的墙时包含CPU预处理与持久化等开销，不是纯CUDA kernel时间。CPU独立审计、任务评分和82.39分钟的首次权重下载不包含在124.58秒内。结果只证明当前固定batch1/分辨率/输出预算可运行，不保证默认最大视觉token或长上下文显存可行。

首次仅做HF和ModelScope各最多8MiB的有界探测。按预锁的最快端到端速度×50%计算，保守完整下载ETA为195.3分钟，原90分钟成本门因此正确停止，未启动全模型下载。root随后在明确资源修订中允许并行后台下载，最多200分钟且07:45硬停；这是资源预算调整，不是重新给原门标通过。

Attempt002复用既有探测、不重复probe，最多两条官方ModelScope流，HF作为有限回退；最终16文件16,595,981,281字节全部校验，五分片首轮成功，失败流0、无bad hash或残留.part。实际耗时82.39分钟，05:37:50完成，含校验有效速度约3.357MB/s。实际快于初始保守ETA，不反向改写旧STOP。所有文件来自公开ungated、Apache-2.0官方模型；权重SHA/size与固定HF清单一致，小文件以固定git-blob/size绑定，没有远程.py、安装或凭据操作。

最终下载manifest SHA：`21c56f351bf1d6bf002f94990a6fe504e63308ecd668d8b291744659c0a2a9b8`。原STOP manifest SHA仍为 `6716bf65ef593b2d920fef0fefc473de1cd63014c52c82d642d452544ad941dd`，原事件、资源修订和两次清单均保留。网络准备不构成方法采纳。

8个合成fixture中，7B命中3个、3B缓存命中1个；它们只描述固定提示和尺寸下的定位响应，不证明真实主体身份或跨平台任务收益。没有据此改坐标单位、校准输出或设置完美命中门，不能用它们为主方法背书。


## 独立审计与封存

独立 CPU 审计 PASS：216 条 token/raw/input-binding 对照、216 次 processor 输入重构、416 个 full/old 映射和全部 1664 个原框精确一致；六个历史控制的任务值独立复现。解析器不执行原回答文本，拒绝重复键、非有限坐标、多对象歧义及超出既定 schema 的指令内容。输入 CPU 重构需要读取图像像素，但没有新增人工语义身份审查或生成模型。

task scoring 只有在审计 PASS、同一 prediction seal 的 root POSTSEAL 授权后执行。独立审计 16.05 秒，任务评分约 0.35 秒；这两个成本不在 124.58 秒模型阶段中。权重审计复用固定下载清单与严格 loader 绑定，不冒称再次独立散列 16.6GB。

有效 generate 显式 do_sample=False、num_beams=1、max_new_tokens=96、repetition_penalty=1.05。保存的基础 generation_config 中 do_sample=True/temperature 字段不代表本轮实际采样，不能混淆有效调用覆盖参数。没有 best-of、技术重答或答案择优。

全部数据是已暴露 A208，Drone2/Quad3 个既有场景，非新确认。场景表保留，不把208条当208独立场景，也不作一般显著性或容量普遍性声明。新增可更新参数为零、没有新3D框、没有源训练、没有TTA；本轮停止当前 point 配置，不改 parser、坐标单位、提示或门槛补救，不读取 U/T/FINAL。

报告与公开数据将保留所有 null、负例、18/6 身份边界、原下载 STOP 与后续资源修订。只公开自有代码、协议、匿名任务标量/状态与模型文件校验元信息；不公开权重、tokenizer 文件、第三方源码快照、token IDs、原回答、原 caption、坐标、mask、点云、真实场景路径或媒体。

证据入口：`../independent_audit/{AUDIT_RESULT,SCORING_RECEIPT,TASK_SUMMARY,COVERAGE_UNKNOWN_SUMMARY,IDENTITY24_SUMMARY}.json`、全部逐例结果/负比较与匿名场景表；原运行、下载和预算证据见相应封存 SHA。
