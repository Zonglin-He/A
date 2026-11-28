# 论文正文精简公式稿

本文使用已登记的 `ray_equivalent_extent_2d_rank`。给定 RGB、语言、LiDAR 与标定，冻结 Qwen 用两次读出得到二维窗口、预测类别及原生旋转框。用标定投影把实测回波与该窗口关联。在局部姿态坐标下以如下五组公式描述主路径。数字门槛、类别准入、相机畸变和回退列在实现段。 [正式登记](../../QWEN_CURRENT_METHOD.json)

## 1. 物理实例绑定

$$
P_o=\underset{R_j\in\mathcal R_{\mathrm{valid}}}{\arg\max}\; \frac{h_j^2}{e_j}.
$$

四尺度地面过滤连通域构成区域候选。$h_j$ 与 $e_j$ 分别为窗内和全图的关联 Fit 回波数。候选须有窗内回波，并有至少五个中央半窗关联回波。校准坐标转换、齐次相机投影和 ROI 集合定义用同一个投影符号 $\pi_{\mathcal K}$ 表示。

## 2. 参考框与轴区间

$$
I_j(a_j,D_j,\eta_j)=a_j+\frac{D_j}{2}[\eta_j-1,\eta_j+1],
\qquad c_j=a_j+\frac{\eta_jD_j}{2}.
$$

PCA 和冻结 Qwen 给出两个姿态。以物理支持轴区间 $[L_j,U_j]$ 构造各自参考框。水平锚取靠光心的端点，光心投影位于区间内时取中点。竖轴取支持和可用地面估计的较低边界。固定下界、上界、中心时分别取 $\eta_j=+1,-1,0$。定义区间已给出由锚和尺寸还原中心的方式。用一句话交代合法支持的 min/max 与 PCA 主轴、长短轴补偿即可。

## 3. 三轴观测等价投影

给定一个参考框 $B_\theta^0$，在其坐标系中合并框内实测支持点与已命中的 Selection 首交点，记为 $\mathcal W_\theta$。此时的锚定模式根据光心相对于参考框的位置确定，竖轴固定底面。$D_{j,\min}$ 是固定该锚并保持 $\mathcal W_\theta$ 在框内的最小跨度，数值实现带 $10^{-8}$ 米保护余量。冻结先验 $D_j^Q$ 取原生旋转框的内在尺寸，最接近竖轴的尺寸用作高度。

$$
D_j^*=\operatorname{clip}\!\left(D_j^Q,\,D_{j,\min},\,D_j^0\right).
$$

把 $D_j^*$ 代回轴区间公式更新中心。$D_{j,\min}$ 的分支放在实现附录。对单边锚，它是保护点相对边界的最大内向距离再加保护余量。对中心锚，它是最大绝对偏移加保护余量的两倍。参考跨度上界保证收缩结果位于原框内。保护下界与上界在容差内相遇时保持参考跨度。

## 4. 保证与适用范围

$$
\mathcal W_\theta\subseteq\widehat B_\theta\subseteq B_\theta^0
\quad\Longrightarrow\quad
\mathcal M_{\mathcal S}(\widehat B_\theta)=\mathcal M_{\mathcal S}(B_\theta^0).
$$

子集关系保留漏击并排除更早的首交，原首交点被保留则保证已命中射线仍在原位置相交。等价关系相对于同一姿态的参考框和有限 Selection 射线。独立 Audit 和跨姿态替换另做测量。固定活动面下的局部秩上界放在理论段，配合 R4 的误差反例。

## 5. 跨姿态选择

令 $k_r$ 为按截断射线残差及历史容差得到的默认姿态，$s_k=\mathrm{IoU}_{2D}(\pi^{\mathrm{AABB}}_{\mathcal K}(\widehat B_k),b_Q)$。$\pi^{\mathrm{AABB}}_{\mathcal K}$ 包括标定八角点投影、轴对齐外接矩形和图像裁剪。无效投影的有效分数 $\widetilde{s}_k$ 设为零，其余为 $s_k$。两候选的选择等价写为

$$
k^*=\begin{cases}
\operatorname*{arg\,max}_{k\in\{\mathrm{PCA},\mathrm{Qwen}\}}\,\widetilde s_k,&\widetilde s_{\rm PCA}\ne\widetilde s_Q,\\
k_r,&\widetilde s_{\rm PCA}=\widetilde s_Q,
\end{cases}\qquad B^*=\widehat B_{k^*}.
$$

射线残差的封顶及未命中惩罚可在实现段用文字和参数表说明。两姿态拟合所需的 Selection 首交仍要计算。

## 原稿的合并位置

| 原位置 | 正文改法 |
| --- | --- |
| §0 输入、7-DoF 及算子组合 | 一段定义，省去 RGB 张量形状、语言 token 形状和三次重复的输出参数 |
| §1 两次 Qwen、相机坐标转换与 ROI | 一句说明两次冻结读出，投影写作 $\pi_{\mathcal K}$。区域得分和它的乘法拆解只留一次 |
| §2 PCA 展开与锚定推导 | PCA 的协方差、特征分解、反正切移附录。保留轴区间与中心更新。说明初始支持锚与投影参考锚的模式各自确定 |
| §3 射线和局部秩 | 只在理论段写首交方程与秩界，删除重复的参数联动例子和假设框序列 |
| §4 可行域、投影与证明 | 留一个 `clip` 公式及候选内蕴含式。平方和展开、`Proj` 等价式和逐行不等式压缩为文字或附录 |
| §5 RGB | 用一个校准外接框算子，保留射线默认与严格增分规则 |
| §6 与 §8 | 合成一张流程图或直接删除重复公式 |
| §7 回退 | 正文一小表，H 尺寸可写 $D^H=\max(s,D^{Q,\mathrm{upright}})$，A 的窗口中位深度与前面补偿放在附录 |
