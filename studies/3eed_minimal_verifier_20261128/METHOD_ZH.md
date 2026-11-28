# S5-pass · Relative Geometric Utility

论文主方法由 **物理假设生成**与**相对几何核验**组成。冻结的 Qwen3-VL-4B 对同一原图和指代做两次独立读出，提供首次二维框及三维尺寸、姿态先验。学习只发生在 Waymo 源域的小型 Verifier。 [论文主方法登记](../../QWEN_PAPER_METHOD.json) · [方法锁](S5_PASS_METHOD_LOCK.json)

## 物理假设生成

Bind 是无可学习参数的点云聚类与区域筛选。在滤除近地面点后，用一张 8 近邻图和四个全局分位数阈值得到嵌套组件。将回波关联到组件，然后按 $h_R^2/e_R$ 取前三，其中 $h_R$ 和 $e_R$ 是窗内和全图的关联 Fit 回波数。 [区域生成消融](../3eed_adaptive_bind_20261128/REPORT_ZH.md)

候选集合以冻结的 Minimal-D 输出 $B_0$ 为第零项，覆盖 C_ray、H、A 和缺失三维输出的回退。候选按几何去重，最多 13 框。

| 来源 | 最多保留 |
| --- | ---: |
| 原始 Minimal-D 锚框 | 1 |
| Top-1 区域的 PCA、PCA+90°、Qwen、Qwen+90° 各自投影后的框 | 4 |
| 原绑定分数第 2、3 名区域各两种姿态 | 4 |
| PCA 与 Qwen 各用 0.85× 和 1.15× 内禀先验尺寸 | 4 |

姿态族相对于自身参考框做逐轴观测等价投影。区域和尺度候选提供不同的物理解释。13 框入口在构造时省去分位数锚假设，保留参考框计算所需的中间量。

## 相对几何核验

每框使用 14 维节点特征，每对候选使用 8 维关系特征。节点包含归一化点云分布、射线命中与残差、首次二维投影及 Qwen 先验对齐。边包含三维 IoU、局部归一化中心差、对数尺寸比与轴向角差。S5-pass 使用三层四头边注意力聚合候选关系。

令 $y_i=\operatorname{IoU}_{3D}(B_i,B_{\mathrm{GT}})$。训练用风险调整效用 $U_i$，推理用相对评分 $u_i$。

$$
U_i=y_i-y_0-0.10\,\mathbf 1[y_0>0.5,\ y_i\le0.5],\qquad
u_i=v_i-v_0.
$$

$$
\mathcal L=D_{\mathrm{KL}}\!\big(\operatorname{softmax}(U/0.05)\,\|\,\operatorname{softmax}(u)\big).
$$

候选排序和替换使用同一个分数。最高分的有效非锚候选只有当 $u_i>0.09659984707832336$ 才替换 $B_0$。阈值和第 12 轮 checkpoint 由退化 Waymo val 冻结。

## 论文结果

| 方法 | 开发 665 条 mIoU / Acc@50 | Mono 148 条 mIoU / Acc@50 |
| --- | ---: | ---: |
| Minimal-D | 34.31% / 35.19% | 34.44% / 37.16% |
| Verifier v2 | 37.84% / 39.55% | 39.31% / 42.57% |
| **S5-pass** | **38.10% / 40.15%** | **39.22% / 42.57%** |

S5-pass 有 **428,041** 个训练参数。相对 v2，开发面板两项差值为 +0.26 和 +0.60 pp，两项录制配对 95% 区间均包含零。Mono 的差值为 −0.09 和 0.00 pp。两个面板均为历史复测，训练和阈值标定只用 Waymo。 [配对消融](REPORT_ZH.md) · [聚合结果](results.json)

使用默认论文 CLI，或用 `registered_predictor(read(PAPER_REGISTRATION))` 载入冻结入口。正式入口已在 850 条缓存预测上逐框匹配既有结果。 [复现命令](../../tools/3eed_recovery/README.md) · [验收回执](S5_PASS_PROMOTION_VERIFICATION.json) · [源码](s5_paper_entry.py)
