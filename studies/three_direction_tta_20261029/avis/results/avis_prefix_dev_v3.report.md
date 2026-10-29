# avis_prefix_dev_v3

已完成固定便利样本的原模型、无更新对照和预声明TTA/音频诊断。以下为本研究面板分数，不是完整AVISeg复现或泛化结论。

| arm | mAP | HOTA | FSLA | reference | ΔmAP | ΔHOTA | ΔFSLA | model s/video | peak GiB |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| source_0 | 36.88 | 18.41 | 35.2 | source_0 | 0.0 | 0.0 | 0.0 | 0.2044 | 4.662 |
| views_0 | 36.88 | 18.4 | 35.2 | source_0 | 0.0 | -0.010000000000001563 | 0.0 | 0.2248 | 4.662 |
| entropy_1 | 36.88 | 18.4 | 35.2 | views_0 | 0.0 | 0.0 | 0.0 | 0.2634 | 4.670 |
| entropy_3 | 36.88 | 18.4 | 35.2 | views_0 | 0.0 | 0.0 | 0.0 | 0.2866 | 4.670 |
| consistency_1 | 36.88 | 18.4 | 35.2 | views_0 | 0.0 | 0.0 | 0.0 | 0.2787 | 4.670 |
| consistency_3 | 36.88 | 18.4 | 35.2 | views_0 | 0.0 | 0.0 | 0.0 | 0.3234 | 4.670 |
| zero_features_0 | 36.88 | 18.4 | 35.2 | source_0 | 0.0 | -0.010000000000001563 | 0.0 | 0.2058 | 4.662 |
| cyclic_audio_0 | 36.88 | 18.4 | 35.2 | source_0 | 0.0 | -0.010000000000001563 | 0.0 | 0.2037 | 4.662 |

TTA比较采用相同双视图但无参数更新的views对照；音频诊断比较原source。每视频正、负、零变化与未定义项完整保存在RESULTS.json。分类/一致性loss下降、自信度或预测稳定都不充当正确性。

覆盖：8视频，每臂128帧。选择为官方压缩包顺序的首8个视频，存在便利采样限制。

时间只覆盖模型前向/更新并含CUDA同步，不含输入加载、缩放、模型初始化和结果编码；每视频每臂单次，不能作为严格部署延迟测量。

FSLA uses the official resolution-dependent empty-RLE convention; no silent scorer fix; zero-support strata are null

决策：不采用本轮TTA作为部署方法。需要独立评分复核、足够的发声切换/同类多实例支持，以及相应代表性数据后再评价机制。

## 评价器审计与独立诊断

原生FSLA保留了两个已知问题：用分辨率特定的PPTl0判断空mask，以及部分ID排列下single-source匹配索引可疑。area-only敏感性只改两处空mask测试，不能称为完整修复FSLA。HOTA敏感性只过滤空面积的预测检测，保留native匹配和26类macro。

| arm | native HOTA | area-zero HOTA | native FSLA | area-only FSLA |
|---|---:|---:|---:|---:|
| source_0 | 18.41 | 18.734794 | 35.2 | 35.197368 |
| views_0 | 18.4 | 18.722871 | 35.2 | 35.197368 |
| entropy_1 | 18.4 | 18.722871 | 35.2 | 35.197368 |
| entropy_3 | 18.4 | 18.722871 | 35.2 | 35.197368 |
| consistency_1 | 18.4 | 18.722871 | 35.2 | 35.197368 |
| consistency_3 | 18.4 | 18.722871 | 35.2 | 35.197368 |
| zero_features_0 | 18.4 | 18.728312 | 35.2 | 35.197368 |
| cyclic_audio_0 | 18.4 | 18.728312 | 35.2 | 35.197368 |

另一份独立审计采用逐帧同类别完整集合匹配正确率，在19个IoU阈值上平均，空面积mask记为不存在；其定义不同于native FSLA。
- consistency_1: 38.980263% (128 frames)
- consistency_3: 38.980263% (128 frames)
- cyclic_audio_0: 38.980263% (128 frames)
- entropy_1: 38.980263% (128 frames)
- entropy_3: 38.980263% (128 frames)
- source_0: 38.980263% (128 frames)
- views_0: 38.980263% (128 frames)
- zero_features_0: 38.980263% (128 frames)

本面板GT只覆盖9/26类别，native HOTA仍对全部26类别macro，绝对值不能直接对比论文完整测试集。三种评价读出均未显示本轮TTA有实质收益；未采用这版配置。

无GT反馈审计发现6个完整视频长度为6帧窗的整数倍，联合时间反转一致性接近浮点数值零，非整数倍两段受padding边界影响更大。仅后期6144个音频融合LayerNorm参数更新，早期frame-query音画定位被冻结；这组配置的近零结果不能否定其他更新接口或反馈信号。

音频幅度条件：官方cached VGGish与发布脚本处理PCM24并/32768的结果一致；改为物理归一化[-1,1]显著改变嵌入，但旧checkpoint在原条件上训练，不能直接宣称修正幅度会改善。原图约1FPS，细粒度嘴唇同步反馈并不自动可用。

GT来自官方gzip尾部的独立可重放恢复，两个DEFLATE block起点、全00/FF字典与tar header/payload一致性已验证，payload SHA256为bec4204121fe404b26ddf8383117b2c9c0aecbd0d7d56b2f0b691b2aaa9a1a90。整包下载及原流同SHA/完整gzip CRC尚待最终回执，不能写成整包已完成。
