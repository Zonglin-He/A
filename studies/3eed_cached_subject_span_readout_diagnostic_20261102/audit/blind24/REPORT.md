# R48：固定24条文字主体head的封后盲对照

**在这24条固定样本上，现有CURRENT代理与人工判断的文字主体head相容；新parser没有提供额外的有效mask变化。官方类别词典跨度反而有5条不落在人工判断的主体head。** 这只是文字角色诊断，不是专家gold、视觉实例身份准确率、GT框匹配或任务性能结论，不能外推整个336条的主体正确率。

## 固定抽样、盲性和未改变的原注释

仅从旧R24 ALL336的caption抽样，platform × 显式Object-type/freeform四层各6条，按sha256(`R48blind|`+id)最低6固定。四层均足额，无补挑。独立主体head与半开字符跨度在算法输出解盲前一次记录并封存，原注释SHA始终为 `5fa0512998f1372e89cd3dc4b32ee21129cf53ad0195f7e6aa1aa87c04e92266`，没有据算法或任务结果改注释。

注释者有历史项目暴露，不声称每条caption全历史未见；本次只对当前R48算法输出和目标标注盲核。24条文字本身的 intended subject head均可判断，没有人工UNKNOWN或多个文字主体候选。文字角色清楚不等于真实场景能唯一定位该实例。

## 两种指标必须分开

- **nominal字符跨度精确相同**：只在方法实际输出原caption跨度时评价。相同词出现在别处不能算同一主体。
- **effective token mask精确覆盖人工head**：对封存的四个mask，逐项使用确切模型输入的actual_offsets检验。UNKNOWN回退到CURRENT也纳入effective运行时行为，但不改记为parser自己找到了head。

官方LITERAL是将另一tokenization上的索引直接用于当前token序列，不能把它的原官方字符跨度当成已经对齐。它仅按实际mask与actual_offsets评价；REMAP另按原charspan投到实际token后评价。native bridge没有独立文字mask，24条全部标记N/A，不当作第五种主体跨度方法。

| 方法 | nominal精确head | effective精确head token mask | UNKNOWN／回退 |
|---|---:|---:|---:|
| CURRENT | 24/24 | 24/24 | 0 |
| text-only parser | 22/24 | 24/24 | 2 |
| official LITERAL实际offset覆盖 | N/A | 19/24 | 0 |
| official charspan REMAP | 19/24 | 19/24 | 0 |
| native bridge | N/A | N/A | N/A |

parser独立找到22条，其余2条回退CURRENT后得到正确的effective mask，不能写成“parser独立24/24”。人工歧义0与算法UNKNOWN2是不同统计。此次没有“同一head词、不同位置仍被记对”的情况，也没有选择sentinel位置当主体。

## 固定分层结果

| 平台／文本格式 | n | CURRENT effective | parser nominal / effective | official LITERAL / REMAP effective |
|---|---:|---:|---:|---:|
| Drone显式Object-type | 6 | 6 | 6 / 6 | 6 / 6 |
| Drone freeform | 6 | 6 | 5 / 6 | 2 / 2 |
| Quad显式Object-type | 6 | 6 | 6 / 6 | 6 / 6 |
| Quad freeform | 6 | 6 | 5 / 6 | 5 / 5 |

官方GT类别引导的同义词列表优先级不等于主体语法角色；它可以先找到描述中的另一种类别近义称呼，而没有指中句首主体head。这里的5条角色不相容全部在freeform层。类别跨度是协议提供的信息，不自动成为主体head金标；同样，本文人工head也不认证数据集目标实例或框身份。

## 访问范围与裁决

本比较只读取原24caption/注释、已封存parser与cases，以及NPZ中的 `actual_offsets`、`positive_masks`、`arm_names` 三个数组。没有加载分数、query、框、GT类别、GTbbox、IoU、metadata、FINAL、媒体、XYZ或新的NLP/模型，也没有据task score筛样本或选指标。

结论限定为：这组固定文字盲核不支持“把CURRENT改成该parser就会修好主体跨度”；官方诊断跨度也不能当语义主体gold。所有原注释、UNKNOWN与不一致均保存；结果不用于本轮继续修parser或反选方法。解析规则和336条预测保持原封存。
