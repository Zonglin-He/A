# Bind、Project、Select

输入为图像 $I$、查询 $T$、LiDAR 点云 $P$ 和标定 $\mathcal K$。冻结 Qwen3-VL 的两次独立读出提供首次二维框 $b_Q$ 以及旋转框给出的姿态与尺寸先验。正文可用两个模块和三个公式描述。

## 实例物理证据绑定

$$
P_o^* = \underset{R\in\mathcal R_{\mathrm{valid}}(P,b_Q,\mathcal K)}{\arg\max}\;\frac{h_R^2}{e_R}.
$$

$h_R$ 与 $e_R$ 分别是关联到候选区域的窗内和全图 Fit 回波数。有效区域还需要中央窗支持达到冻结门槛。区域由跨尺度的地面过滤连通域给出。绑定在 Fit 回波上决定实例，按保留像素单元携带 Selection 射线。Audit 只用于事后检查。

## 观测等价几何恢复

$$
\widehat B_\theta = \underset{B\in\mathcal E^{\mathrm{axis}}_\theta}{\arg\min}
\;\lVert D(B)-D^Q\rVert_2^2,
\qquad \theta\in\{\theta_{\mathrm{PCA}},\theta_Q\}.
$$

每个姿态先由 LiDAR 支持近面、可用地面和直立外包跨度确定参考框 $B^0_\theta$。锚 $a_j$、模式 $\eta_j$ 和跨度 $D_j$ 表示轴区间 $a_j+\frac{D_j}{2}[\eta_j-1,\eta_j+1]$，中心是区间中点。参考框由支持区间延展得到。投影锚按光心相对参考框的位置确定，竖直轴固定底面。支持锚和参考框锚各自确定。

$\mathcal E^{\mathrm{axis}}_\theta$ 由该姿态固定轴锚后的跨度区间构成。其下界保留参考框内的实测支持点和原有 Selection 首交点，上界维持 $B\subseteq B^0_\theta$。逐轴把原生尺寸先验 $D^Q$ 截断到区间并还原中点即可，代码保留历史容差及无有效首交时的参考框。嵌套约束阻止首交变早或漏击变为命中，保护原首交点阻止命中变晚或变成漏击。该保证针对固定候选的参考框和它的 Selection 射线。

$$
\theta^*=\begin{cases}
\operatorname*{arg\,max}_{\theta}\widetilde S_{\rm RGB}(\widehat B_\theta), & \text{两候选有效分数不同},\\
\theta_{\rm ray}, & \text{有效分数相同},
\end{cases}
\qquad B^*=\widehat B_{\theta^*}.
$$

$\widetilde S_{\rm RGB}$ 是所恢复框的裁剪二维投影外接框与 $b_Q$ 的 IoU，无效投影记零。一方的分数更高时直接选中。平局沿用参考框的截断射线损失、未命中罚项和原容差。拟合已得到参考首交深度，因此平局打分可直接复用。

H 与 A 在证据不足时作为回退表描述。H 用可用物理区域和先验完成框。A 用二维窗口视线及可用 LiDAR 中位深度锚定原生框，二维回复缺失时直接读出原生框。三维回复缺失时无预测。类别准入、校准和固定门槛列在实现细节中。 [重构实现](minimal.py) · [实验协议](PROTOCOL_ZH.md)
