# HC2 顺序坐标调参实际结果

最终配置：`{"lr": 0.01, "sigma": 0.025, "tau": 0.05, "steps": 40, "writeback": 0.0625, "samples": 32}`。按 lr→sigma→tau→steps→writeback 一轮顺序锁定；26次逻辑候选、22套完整配置、0套数值无效配置。完整配置共1408条开发到达，原32父来源×2顺序全部纳入。

选定配置开发 parent-macro ΔvIoU=3.572563pp，10000 paired source-bootstrap 95% CI=[1.285352,6.720775]pp；current=2.717506pp，inherited=0.855057pp。原起点开发 ΔvIoU=2.733575pp。负尾超过5/20pp的来源为0/0，全部来源和负结果在 ROOT_AUDIT 与 ALL_CANDIDATES 中保留。

独立统计/链路核验4405项通过；每条完整预测已复核 Gaussian likelihood、detached reward、Adam 算术、query 重置、1792参数状态链接、LN写回和末轮输出，以及官方和独立 dense v/t 指标。未独立重放解码器 Jacobian。

数据是历史曝光的 HC2 validation 开发来源，所得分数有选参偏差；128来源确认集和P1全量分数未参与此次选择。不能把开发置信区间称为独立确认，也不能以此宣称HC2稳定提高。仅一轮预设网格内条件最优，可能受参数次序与交互影响。

成本分GPU适应、冻结输入capture和CPU数学审计；复用精确旧输入/历史配置的成本标明为原测量，不能声称冷启动端到端速度。VidSTG参数和算法结构保持原样，EATA暂停。旧P1 685条原预测保持；参数变化后在独立修订目录重新锁定从源模型开始的HC2完整流，不拼接不同参数的prefix。
