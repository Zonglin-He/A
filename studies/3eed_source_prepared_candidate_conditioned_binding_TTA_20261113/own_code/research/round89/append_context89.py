"""Append retained historical benchmark, measured soft-support and private review links."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text());text=(H/'REPORT.md').read_text();ss=read(H/'HISTORICAL_ABC_SUMMARY_ROWS.json');cov=read(H/'SOFT_SUPPORT_COVERAGE_ROWS.json');vv=[r['head_averaged_proxy_soft_support_share_mean'] for r in cov if 'head_averaged_proxy_soft_support_share_mean' in r];text+='\n## 保留的历史三采样基线\n\n除本轮29臂外，原Frozen ABC直接复用R88在相同336条上的封存结果，不运行新模型、不改方法或源域选择。新增全体逐条与全部29臂对比均保留，包含所有负差。\n\n| 平台 | n | Frozen ABC mIoU % | 主要方法−ABC pp | 改善/恶化 |\n|---|---:|---:|---:|---:|\n'
for r in ss:text+=f'| {r["platform"]} | {r["n"]} | {r["frozen_ABC_mIoU_percent"]:.6f} | {r["candidate_tta1_gain_pp"]:+.6f} | {r["candidate_tta1_better_n"]}/{r["candidate_tta1_worse_n"]} |\n'
text+=f'\n## 局部读取范围的额外边界\n\n本轮沿用0.05全局底权重，并非严格实例mask。基于头平均原生注意力估计的候选自身支持内质量比例，先对每条候选取均值、再对样本取均值，为 {float(np.mean(vv)):.6f}。这是头平均输入的描述性代理，并不等于逐head归一化后实际向量的精确占比，更不是实例纯度。它保留了原方案的全局安全通路，也说明“候选专属局部向量”不能被解释为只含该物体。后续不能把这个实现的阴性结果推广到所有局部语义绑定结构。\n'
text+='\n本地案例图册与人工查看记录：`crossplatform/round89/atlas_PRIVATE/index.html`、`research/round89/CASE_REVIEW.md`。它们仅用于封存后的机制解释，不进入方法选择。公开包不含原句、原图或私有向量。\n';(H/'REPORT.md').write_text(text)
