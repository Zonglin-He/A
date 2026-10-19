# Root route decision

真实DINO下没有建立独立OPD收益：Live−E完整v HC−0.3974pp、Vid+0.1032pp，未观察actual s两域负；Live−Direct完整v−0.2644/−0.0065pp、Live−Fixed−0.3772/+0.0555pp，比较区间均跨零，未达到所需2–3pp量级。故不采用此E保留1792＋固定框IoU坐标OPD组合，结束这条具体主线，不扫描参数或追加观测。

正确监督能力证据仍保留：GT-Live相对E的HC完整v+4.8221pp CI[+1.3626,+9.2611]，未观察actual s+3.2671pp但CI跨零；GT-Direct在Vid完整v+1.1364pp和未观察actual s+2.8658pp的区间均正。HC GT-Live超过GT-Fixed的完整v+4.5623pp区间正，但超过GT-Direct尚不明确；它支持在正确证据下刷新可能有用，不能替代真实证据结果。

这不是“1792入口没有学习能力”或“OPD普遍无效”。真实参考与GT正控支持53→43/35→16，省略未知位置仍是归因边界；且正确参考下GT-Direct的NH007仍完整v−18.6932pp，未观察传播也会损伤，不能把全部失败唯一归为教师。真实Live的NH007−20.2214pp严重负尾原样保留。当前证据只够停止本连接配置，原E＋MBR/Fine研究基线和正式评估/生产注册保持，不自动启动下一轮。

Real and GT controls separate; differing unknown support disclosed. All outcomes including severe negatives retained. No automatic follow-up, checkpoint or production promotion.

另外，GT正控NV012的Live连续10轮奖励全零而不更新，Direct却完整v+6.3720pp；HC的NH001/NH005各有40轮GT-Live全零奖励，Direct分别+2.3346/+1.1146pp。这些是同一正确参考支持下的实际漏学反例，不能把全部差距归为教师证据，也不能把平奖励跳更新当作可靠性判断。
