# R29审计数值修订001：原模型锁和预测保持不变

2026-10-20，任何本轮新GT之前。原128父七输出测量已完整封存；原PLAN、93pins（包括原scorer/publisher/protocol）、GLOBAL模型barrier及预测/Source控制raw收据全部保留原字节。只更正独立数值oracle，无新模型/编码/学生更新/预测，不改变候选、图像、空间、误差门槛、统计或研究准入条件。

原scorer在RC056/offset1预GT检查失败，插值差2.02916e−6略超原16eps32×max(1,|reference|)。局部角点幅度16.484375而输出约.515，坐标舍入能被高梯度/相消放大。NumPy u/v提前舍入最初只是猜测，诊断中old/double u/v相同，不能写成原因。纯FP64数学坐标oracle仍失败。

本机torch2.7.0+cu128，git134179474539648ba7dee1317959529fbd0e7f89。GridSampler.h的scalar实现((coord+1)*size−1)/2复原RC056，但全量预审计又在RC081失败1.97254e−6。进一步核对同commit官方CPU GridSamplerKernel.cpp:267–275的vector表达(in+1)*(size/2)−.5及本机AVX融合运算。独立CPU斜坡探针确认RC081实际x=1.9499999284744263，分开舍入scalar式=1.9500000476837158，contracted式精确匹配。旧失败/草稿/诊断全保留，不能把scalar头文件误称整个CPU后端。

原文链接：https://raw.githubusercontent.com/pytorch/pytorch/134179474539648ba7dee1317959529fbd0e7f89/aten/src/ATen/native/cpu/GridSamplerKernel.cpp

新oracle从保存FP32 grid计算FP32(in+1)，随后用FP64精确中间值计算乘size/2减.5再一次round至FP32，模拟本机融合乘加坐标；border/floor后独立FP64 bilinear/alpha混合。称“本机CPU vector contracted FP32坐标语义＋FP64独立插值”，不称全FP64数学坐标或其它硬件恒等。不调用torch grid_sample当expected。原注册16eps32×max(1,|reference|)、local2eps16和Native bitexact门槛均不改；FP16 local和FP32 registered最大误差分账，不能用前者数值解释后者门槛。

全部128无GT预审计pass：204800独立插值标量、578305几何格、512空间/384旧输出/384CDF、12532 MBR抽点和raw Source控制通过。预审计以SystemExit明确停在GT之前，退出码1是这个无GT预览的有意停止，不是任务评分成功。未产生任何GT_EXPOSURE。

实际恢复入口scripts/score_stvg_temporal_registered_motion128_audit001.py及发布入口scripts/publish_stvg_temporal_registered_motion128_audit001.py另pin。SCORING_PLAN_LOCK001绑定原PLAN SHA、原模型barrier SHA及新增审计代码/本修订/源代码材料SHA；原93pins仍全部核验，模型barrier仍绑定原PLAN。恢复评分将再次完成全部预审计再读GT，资格/统计完全不变。原scorer保留为原失败复现入口。

公开仅数值公式/摘要/GT前修订时序，不公开原特征向量、私有张量/媒体/文本/ID。成功恢复评分CPU时间单列；额外失败/诊断/预审计CPU wall未逐次精确计时，明确不能当作零，也不混称成功入口时间为全部成本。新增GPU/模型/专家/空间或学生更新全部0，原模型GPU成本不变。
