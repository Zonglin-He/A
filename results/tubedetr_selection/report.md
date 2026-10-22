# 固定候选评分与准入：两域128完整验证

候选坐标、时间分支、Source、教师、八观察和输入不变；只修改评分/准入。全部预测先封存后独立CPU评分，没有新神经网络调用或全量。所有负配置保留。128已重复曝光，以下不是独立泛化确认。

| 面板 | 配置数 | Full dense | dense增量pp | 未观察dense | 未观察增量pp | Full v | v增量pp | 两项空间均达+.1pp |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| vidstg128 | 12 | 48.696971 | +0.080552 | 48.673248 | +0.081603 | 19.743063 | +0.069309 | False |
| hc2128 | 10 | 57.727213 | +0.130526 | 57.696266 | +0.133564 | 27.731566 | +0.082021 | True |

增量相对同一Source+Global6（Temporal）；另补同初始化ERef-only控制，控制在原trial终态后单独冻结，不用于重新选择配置。

## vidstg

评分参数：`{"selection_mode": "cross_agreement", "margin": 0.0, "cycle_floor": 0.1, "agreement_floor": 0.8, "guard": 2.0}`；候选参数固定：`{"gamma": 0.0, "admission": 0.4, "guard": 2.0, "link": 1.0, "mode": "relative_residual", "alpha": 1.0, "beta": 0.5}`。点估计护栏=True，实际增量目标=False，两项空间bootstrap下界均正=True。

相对Temporal：s +0.080552pp，95%父bootstrap[0.010406,0.165295]，胜/负/平18/8/102，逐父留一最低+0.060183pp；unobserved +0.081603pp，95%父bootstrap[0.009992,0.168072]，胜/负/平18/8/102，逐父留一最低+0.060664pp；v +0.069309pp，95%父bootstrap[0.001278,0.151017]，胜/负/平18/7/103，逐父留一最低+0.049128pp。

相对ERef_only：s +0.080552pp，95%父bootstrap[0.010406,0.165295]，胜/负/平18/8/102，逐父留一最低+0.060183pp；unobserved +0.081603pp，95%父bootstrap[0.009992,0.168072]，胜/负/平18/8/102，逐父留一最低+0.060664pp；v +0.069309pp，95%父bootstrap[0.001278,0.151017]，胜/负/平18/7/103，逐父留一最低+0.049128pp。

相对original_score：s +0.114515pp，95%父bootstrap[-0.457893,0.644302]，胜/负/平35/53/40，逐父留一最低+0.012815pp；unobserved +0.032165pp，95%父bootstrap[-0.470680,0.476359]，胜/负/平34/47/47，逐父留一最低-0.051215pp；v +0.043181pp，95%父bootstrap[-0.194040,0.274718]，胜/负/平34/53/41，逐父留一最低-0.009071pp。

相对observation_protection：s +0.016398pp，95%父bootstrap[-0.010002,0.044747]，胜/负/平16/10/102，逐父留一最低+0.008248pp；unobserved +0.017321pp，95%父bootstrap[-0.009666,0.046274]，胜/负/平16/10/102，逐父留一最低+0.009088pp；v +0.011697pp，95%父bootstrap[-0.010301,0.035897]，胜/负/平15/10/103，逐父留一最低+0.004443pp。

实际物理节点（独立于dense插值统计）：{"GT_annotated_unobserved_nodes": 6232, "pooled_node_IoU_gain_vs_Source_pp": 0.17178174086515302, "equal_query_node_IoU_gain_vs_Source_pp": 0.08457082513915165, "improved_nodes": 297, "harmed_nodes": 112, "actual_unobserved_actions": 746, "actual_observed_actions": 0, "new_admitted_nodes": 1901, "original_eligible_nodes": 4400, "scope": "Physical GT-supported unobserved nodes; not dense interpolated estimand. Source/ERef differences on HC include fixed initialization."}

## hc2

评分参数：`{"selection_mode": "observe_protection", "margin": 0.0, "cycle_floor": 0.1, "agreement_floor": 0.0, "guard": 0.4}`；候选参数固定：`{"gamma": 0.025, "admission": 0.7, "guard": 0.4, "link": 1.0, "mode": "original"}`。点估计护栏=True，实际增量目标=True，两项空间bootstrap下界均正=False。

相对Temporal：s +0.130526pp，95%父bootstrap[-0.007747,0.317283]，胜/负/平75/53/0，逐父留一最低+0.060273pp；unobserved +0.133564pp，95%父bootstrap[-0.006901,0.323960]，胜/负/平76/52/0，逐父留一最低+0.061684pp；v +0.082021pp，95%父bootstrap[0.015087,0.177313]，胜/负/平73/48/7，逐父留一最低+0.044296pp。

相对ERef_only：s +0.113865pp，95%父bootstrap[0.000000,0.285865]，胜/负/平3/0/125，逐父留一最低+0.043615pp；unobserved +0.116057pp，95%父bootstrap[0.000000,0.291552]，胜/负/平3/0/125，逐父留一最低+0.044189pp；v +0.058486pp，95%父bootstrap[0.000000,0.145518]，胜/负/平3/0/125，逐父留一最低+0.020644pp。

相对original_score：s +0.096224pp，95%父bootstrap[-0.178071,0.404328]，胜/负/平40/70/18，逐父留一最低+0.020045pp；unobserved +0.124707pp，95%父bootstrap[-0.103350,0.389379]，胜/负/平40/70/18，逐父留一最低+0.054700pp；v +0.012845pp，95%父bootstrap[-0.159405,0.193736]，胜/负/平41/68/19，逐父留一最低-0.027270pp。

相对observation_protection：s +0.000000pp，95%父bootstrap[0.000000,0.000000]，胜/负/平0/0/128，逐父留一最低+0.000000pp；unobserved +0.000000pp，95%父bootstrap[0.000000,0.000000]，胜/负/平0/0/128，逐父留一最低+0.000000pp；v +0.000000pp，95%父bootstrap[0.000000,0.000000]，胜/负/平0/0/128，逐父留一最低+0.000000pp。

实际物理节点（独立于dense插值统计）：{"GT_annotated_unobserved_nodes": 5775, "pooled_node_IoU_gain_vs_Source_pp": 0.11538700028822589, "equal_query_node_IoU_gain_vs_Source_pp": 0.15939813717893442, "improved_nodes": 2952, "harmed_nodes": 1901, "actual_unobserved_actions": 38, "actual_observed_actions": 0, "new_admitted_nodes": 618, "original_eligible_nodes": 1275, "scope": "Physical GT-supported unobserved nodes; not dense interpolated estimand. Source/ERef differences on HC include fixed initialization."}

## 边界

不同锚点的交叉证据仍可能同错，不是身份真值。观察保护关闭的是直接覆盖，HC仍含原gamma初始化；ERef控制用于分解这一点。未观察dense会受到观察节点两侧插值影响，必须与实际未观察节点统计并读。小幅正点估计不能冒充稳定收益；bootstrap在选择后描述且未修正多轮开发选择。旧全量没有改写，本轮不提供自动全量入口。
