# R25 冻结时间模块条件随机推理：预锁

空间方法/config和E/Live41/42/43坐标全部冻结；同TA-STVG Vid Source checkpoint（SHA5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83），无新专家/观测/模型/权重训练/学生或空间update。RC128反复曝光开发父query；TC109不选择配置。目标2–3pp以上完整v的可重复增量仍未达。

旧R14为确定性的两pass/多层分布平均；R23反事实家族和R24节点数条件化配置已关闭，不恢复温度/先验/步骤搜索。新机制只检验：原有Dropout时间模块随机前向积分，能否提供更好的条件预测分布。依据Gal&Ghahramani2016原文https://proceedings.mlr.press/v48/gal16.html及当前实际网络中原生随机模块；论文不保证本checkpoint校准。缺少实际训练调用日志，不能声称验证了完整Bayesian推断或新事件知识。

固定缓存memory/位置/mask/Native第二pass query与gates，不重跑encoder/图像/第一pass query/action gates。只激活最终TimeDecoder六层的12个MultiheadAttention内部dropout和24个FFN/残差Dropout，原rate .1；预测MLP原两次Dropout .3（包括二维logits后的Dropout）。整个模型保持eval/frozen，仅选中叶节点切换training flags，全部恢复；不使用model.train，不触动LN/BN统计。query_content未被TimeDecoder使用、time_fc不在有效路径，不计作用。

唯一FullMC候选：seed41/42/43每流16draw共48，按固定ordinal/offset/draw公式独立种子；每draw先两offset logits→原始物理后验CDF/envelope→原Native正支持条件化（记录排除质量），再平均48完整后验，不均logits/边际后再组合。三流只作积分预算，不能选最佳流或当独立父样本。原2080 MBR动作完全保留。

匹配控制Native、HeadOnlyMC（固定Native decoder hidden，只激活原MLP两次dropout）、OutputOnlyMC（仅原最后logits dropout）、既有强HiResAuxMix。Full和HeadOnly按两次head调用独立重置相同随机种子，显式重建mask验证实际输出，最终OutputOnly使用同一最终mask。Full额外随机解码器；不能把Full对Native获益自动归为新增语义知识或OPD。

原Native前/后logits与后验必须exact。首2父执行selectedrate=0正控exact、同seed完整draw再现exact；实际MHA/Dropout/head调用计数、mask配对/公式、decoder/head flags/rates恢复、权重hash/无梯度均验证。全局起止参数/flags不变；保存全draw的两个offset实际Full/HeadOnly/Output logits、head maskSHA及最终mask/前logits、seed、原native hidden/logits、概率/读出、空间坐标、来源SHA。存热图不算适应，本轮没有学生更新。

GPU单run<=900s（含载入和清理）；先2父全48draw无GT测速，保守估算loading+128*maxcase+60必须<900才继续全128。新Source RGB/encoder=0、专家=0，但时间decoder随机前向完整计费。最多12808 decoder offset calls（Full12288+Native512+首2zero4+replay4），headOnly另12288。非阻塞共享gpu.lock和own run.lock；主机6GiB/磁盘30GiB、GPU启动free4GiB并保留5GiB、allocator<=.8；预算不足保留部分，不能读GT或缩draw救结果。

全128预测、draw及成本全seal/global barrier后才新GT。CPU独立从真实logits重构每draw原CDF与平均、mask相乘/配对、原动作效用、Native/强/空间exact、官方Dense和全2080候选质量；每CPU阶段900s同run.lock及resource/pins前后检查。父内平均3冻结空间seed、10000父bootstrap，全父全负尾/留一/边界/tIoU及完整v，期望GTv单独评价。

唯一FullMC相对Native硬完整v>=2pp且CI正、并超过HeadOnly/OutputOnly/HiResAuxMix才静态确认资格；期望候选GTv同条件>=2pp/CI正及控制胜出才一次实际Direct/Fixed/Live受控转移资格。hard失败不等于OPD失败；输出噪声控制不事后晋升。否则关闭本条件MC配置，不扫rate/draw/种子/混合权重，不改旧结果。完成/失败当轮档案check snapshot check与GitHub public allowlist全signed负结果，dualref freshCAS及远端bytes/hash/tree；CURRENT/MAIN/PAPER不自动晋升。
