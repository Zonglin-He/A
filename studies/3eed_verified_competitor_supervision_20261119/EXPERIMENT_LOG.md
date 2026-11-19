# 执行记录

1. 锁定同一源域样本、同一 D 候选池和 A、B、C、D 分解，核对上轮远端回执和本地原始文件签名。
2. 对原 D、旧成对和旧全池的高分竞争者补查。核验图像隐藏方法和分数。封存 26 个新增参照负例以及 11 个仍未知的候选。
3. 先执行 A、B 三标量对照。按录制复算所有结果，并复查更大的相同求解预算。
4. 重新采集固定候选的冻结特征，精确重放 D 框和分数。用相同 10,240 参数的旧头执行 C、D、关闭 RGB、错配 RGB。
5. 首次头部输出在一个空训练折返回了数组，JSON 拒绝序列化。改为相同数值的列表，保留旧日志和当时封存的标量结果。修复前后的两份标量 CSV 与赢家监督 CSV 逐字节一致。
6. 主种子的 D 出现小幅排序收益。锁定配对替代种子和匹配实际赢家验证，保留两次头部结果。拒识条件与种子稳定性未通过。
7. 独立复核候选三维排除、正例保护、RGB 控制、外层与内层录制划分、排序和拒识汇总。未执行下游联合与目标域。

使用工作区既有 3EED 环境。先运行 `supervise.py prepare`，按盲图完成 `supervise.py seal`。使用 3EED 的模型运行目录执行 `capture_features.py`。按序运行 `compare.py scalar`、`compare.py numerical`、`compare.py heads`、`verify_added_labels.py`、`validate_candidate.py verifier`、`validate_candidate.py seed`、`recording_intervals.py` 与 `audit.py`。数据、图像和训练参数均为本地工件。
