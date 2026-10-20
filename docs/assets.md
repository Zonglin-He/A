# 本地资产

数据、权重、解析器模型、安装环境、编译算子和运行输出均留在本地，不上传 GitHub。

必须保留三个冻结检查点：`checkpoints/TASTVG_VidSTG.pth`、`checkpoints/TASTVG_HCSTVG2.pth`、`checkpoints/referdino/ryt_mevis_swinb.pth`。权重 SHA256 见 assets.json；不会自动下载或用不同权重替换。

Source 构造还需要原 Video-Swin 初始化文件 `checkpoints/tastvg_model_zoo/`、RoBERTa 完整快照 `checkpoints/huggingface/`、ResNet101 V1 文件 `checkpoints/torch/`，以及查询解析器的 `checkpoints/stanza/`。这些是现有冻结模型的构造依赖，不是新增适应专家。

ReferDINO 的 BERT 由同一完整检查点的 bert.* 状态严格导出到 `runtime/referdino/bert-from-frozen-teacher/`。tokenizer 使用已保存的 BERT tokenizer 文件。`tools/setup.py --prepare` 仅准备已有文件；缺少文件时明确报错，不会联网选择新模型。

ReferDINO 的九个额外 Python 包安装到 `runtime/referdino/python/`，与主环境隔离。CUDA 算子位于 `runtime/referdino/cuda_build/`。可重建源码保留在 `external/ReferDINO/compat/`，只含已有 PyTorch2.7 C++/CUDA ABI 修正，没有 decoder 或推理逻辑修改。编译需匹配本机 CUDA toolkit；已验证机器为 RTX5090/CUDA12.8，架构12.0。

`runtime/source/` 保存无标签模型构造信息及 model_zoo 链接；`runtime/gpu.lock` 是共享 GPU 排他锁。旧锁路径保留同 inode 的兼容链接，以免其他已有工具绕过共享租约。锁不是实验缓存。

`docs/assets.json` 提供当前可复现资产的相对路径及校验值。`tools/setup.py --check` 仅读文件检查，不执行模型前向。FFmpeg 的字节指纹与既有解码规则一并保留。安装出不同构建的 FFmpeg 时应先核对输入解码一致性，不能静默跳过检查。

当前机器的主方法目录为 `stvg-opd/`；上述五个私有资产目录通过指向外层的本地链接使用。它们不属于 Git 树，独立克隆可使用真实目录或自行配置本地链接。
