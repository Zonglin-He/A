# 2026-09-28 公开副本验证

本次在公开副本目录、现有本地研究 Python 环境运行以下 CPU 检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_desta3d*.py
```

结果：**71 passed in 3.56s**，23个测试文件。包括真实hidden128模块合成输入、优化器整数key/live Parameter恢复后两次更新、冻结作用域、三步Adam/复位、完整词表FP32 head目标/梯度、缺失和无效几何合同、统计估计量及cache/offload原语。

这些是CPU合同和合成控制；没有加载4B模型、没有读取视频/评分标签、没有GPU重跑。本次也不是在干净虚拟环境重装依赖后的完整可复现性验收。CUDA RNG在CPU恢复测试中mock，不代表真实GPU验证；历史真实GPU验收是独立证据。

发布时另外检查Python语法、入口文档相对链接、文件类型/大小、私有路径/凭据模式以及源码映射。完整GPU入口仍依赖未公开的合法数据、权重和锁定manifest。统计JSON不含逐样本标识/预测；旧历史结果不覆盖。

不同研究阶段的runner存在不同数据合同。不能把所有历史runner当成一套CLI，也不要为了运行而跳过pins、sealed-prediction前置检查或源/目标数据边界。
