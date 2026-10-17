# STVG: DINO P7-A + online Gaussian OPD

当前按用户要求恢复历史DINO版本：Source原生时间I0内获取Uniform4稀疏参考，
Actionness投影给出输出时间I1；1792参数Gaussian OPD更新Query残差和Decoder LN。
Query与Adam逐条重置，LN按原比例跨Query保留。两域原参数保持，不重新调参。

- 主方法入口：`methods.stvg_opd_p7a_v1.predictor.P7AOPDPredictor`
- [当前方法注册](methods/MAIN_METHOD.json)
- [原始方法说明](methods/stvg_opd_p7a_v1/README.md)
- [本次恢复验证协议](protocols/stvg_dino_rollback_verify32_v1.md)
- [历史原始P7-A协议](protocols/stvg_opd_p7a_paper_revalidation_v1.md)
- [验证入口](scripts/verify_stvg_dino_rollback_v1.py)

Qwen官方实验与自动接续已由用户暂停；已完成结果和未完成产物均保留。
本次32历史来源的实现回归验证、完整预测封存与独立评分已完成。
正式全量与其他历史队列尚未重启；[验证报告](results/stvg_dino_rollback/2026-10-17/verification32/REPORT.md)保留完整正负结果。
验证完成状态、全部正负结果与限制以实际ROOT/CPU和远端公开收据为准。
权重、私有媒体、标注、Query/Caption文本、原始响应和缓存不公开。
