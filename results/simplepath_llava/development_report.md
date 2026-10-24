# 新版LLaVA固定方法：双域原128完整记录

固定协议，无选参、性能筛选或换query。原生Source生成时间与框不变；NativeMBR仅额外时间读出控制，不能重命名原生Source。真实逐起点条件时间候选和首STP局部表示资格已通过，当前双域128完整seal及独立评分后进入原总体全量。

| Dataset | Arm | vIoU | dense sIoU | tIoU |
|---|---|---:|---:|---:|
| vidstg | Source | 13.916442 | 27.238891 | 45.983889 |
| vidstg | Spatial | 17.872056 | 32.834148 | 45.983889 |
| vidstg | NativeMBR | 14.681301 | 27.238891 | 48.605634 |
| vidstg | Temporal | 14.580397 | 27.238891 | 48.500443 |
| vidstg | Full | 19.884277 | 33.101206 | 48.500443 |
| hc2 | Source | 9.563491 | 36.572585 | 23.371198 |
| hc2 | Spatial | 12.130855 | 41.217250 | 23.371198 |
| hc2 | NativeMBR | 17.913078 | 36.572585 | 42.168912 |
| hc2 | Temporal | 18.486268 | 36.572585 | 43.710830 |
| hc2 | Full | 20.964430 | 40.992706 | 43.710830 |

## 核验范围

Root首次核验全部256个新预测与receipt SHA、request/plan绑定、有限正尺寸、全部后验从native_prior与frame_feedback重构后的原数组SHA、Source/原生时间/物理网格/八观察对对应旧full逐位一致。所有五行均值独立DenseTube重算通过。只读取对应256条旧参考，不重扫旧12205或16GB权重。两份开发seal与两份metrics的gate引用均核验，总gate mtime早于首full receipt。

完整匿名逐query五行标量及所有正负结果公开保存；原始框/特征/媒体/文本/标注不公开。本报告不据开发反馈改变参数或停跑；同一merged checkpoint训练含Vid衍生指令、开发及此前全量已研究曝光，不称严格未见域或独立未曝光测试。

Full−Source包含新增时间读出与空间路径的总变化；Temporal−NativeMBR是固定同读出下时间反馈控制，空间变化另保留Spatial、Full及observed/unobserved分项。不把全部时间增益归给原型。

当前执行顺序：VidSTG10303/732完整seal→CPU评分→HC1902/227完整seal→CPU评分→Root全量核验与最终发布。尚无本轮完整全量成绩。本开发发布不替代未来final。当前GPU结束后，已锁定的TubeDETR空间搜索全量队列按共享锁串行接续。
