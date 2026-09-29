# DESTA 双专家原生伪监督：读码与调参

本版本把时间专家区间和空间专家轨迹转换为 PTD 原生 action targets，
只更新每个 query 的 R16 latent field。原 PTD/B1 和专家权重冻结。
不使用旧 offline mixer、pixel dim/blur 或 OPD。

建议按这个顺序读：

1. `desta3d/native_adaptation.py`：伪目标映射、完整空间梯度归一化、R16 投影与半径约束。
2. `scripts/desta_native_experts.py`：UniversalVTG、GroundingDINO + SAM2 独立推理和原始证据保存。
3. `scripts/desta_native_run_v3.py`：固定 B1 原生支持、分别反传、更新 C、末态自由解码。
4. `scripts/score_desta_native.py`：先封存后离线评分，按父源宏 vIoU 选配置。
5. `scripts/supervise_desta_native_v3.py`：27→6→1 和最终 T-only/S-only 消融自动执行。
6. `protocols/desta_dual_expert_native_v1.md`：本轮实际完整合同。

## 三个主参数

| 参数 | 本轮取值 | 实际作用 |
|---|---|---|
| `steps` / K | 1, 3, 5 | 梯度重算次数；单步长度为 rho*norm(F)/K |
| `radius` / rho | .03, .07, .135 | latent 总位移上限，相对于原 F 范数 |
| `temporal_weight` | .5, 1, 2 | T/S 完整梯度各自归一化后，时间项的相对权重；空间权重1 |

`AdaptationConfig` 就是每个 arm 的真实配置。`grid_configs()` 生成本轮 27 臂，
登记后的 `CONFIG.json` 保存实际参数。阅读、复制或新建未来实验时可以修改这些值；
正在执行与已经封存的目录和代码 pins 不应原地覆盖。权重、native support、专家阈值
以及R16 basis属于另一层配置，本轮固定，不能从本轮三因子网格推断它们的影响。

## 执行入口

下面命令用于一个**尚未登记的新版本**；当前版本已执行的阶段不可重复启动。

```bash
PYTHONPATH=. .venv-ptd-audit/bin/python -B -m pytest -q tests/test_desta_native_adaptation.py
.venv-ptd-audit/bin/python -B scripts/desta_native_common.py
.venv-ptd-audit/bin/python -B scripts/launch_desta_native.py temporal --run temporal001
PYTHONPATH=.runtime/desta_spatial_deps .venv-ptd-audit/bin/python -B scripts/launch_desta_native.py spatial --run spatial002
.venv-ptd-audit/bin/python -B scripts/supervise_desta_native_v3.py
```

时间专家使用 `.venv-exost`；PTD 使用 `.venv-ptd-audit`；空间 tracker 的轻量依赖
独立放在 `.runtime/desta_spatial_deps`，不改 PTD 安装包。空间模型是检测+跟踪备选，
不是任务训练 RVOS。SAM2 当前缺 optional connected-components CUDA extension，
官方实现跳过 hole-filling 后处理；这一实际运行条件需在报告中保留。

结果目录：`artifacts/desta3d_v3/latent_oracle_v1/dual_expert_native_v1/`。
`scores/dev16/REPORT.md` 是完整27配置表；`scores/dev64/REPORT.md` 是前6扩展；
`scores/ablations/REPORT.md` 是单专家消融。最终参数影响结论见根 `REPORT.md`。
本轮使用已曝光 development GT 离线挑配置，不能作为未见数据泛化结果。

## 当前运行的工程版本

科学更新核心仍是 `desta3d/native_adaptation.py`。当前 worker 为
`desta_native_run_v3.py`，通过 `desta3d/native_runtime.py` 将 host activation
offload 上限设为14GiB，host reserve保持6GiB；原memory-v7代码不改。
这只改变保存张量驻留位置，不改变dtype、数值、stride、loss或梯度。
原v1/v2因为32GiB主机的可用内存不足而触发保护，失败与部分步骤全部保留。
首例两支C0原生logits、loss及完整raw梯度已逐值验证一致。

`continue_desta_native_v4.py --after-run dev16003` 是本轮一次性接续器：
只等待现有wrapper的真实进程句柄，成功退出后运行v3 controller；失败不重试。
用户手动读码时主要看上述科学核心与v3 worker，旧版本是复现/失败证据。
不要再次执行本轮接续器；当前进度看 `ACTIVE.json` 和 `CONTROLLER_V4.log`。

工程修复补充协议：`protocols/desta_dual_expert_native_runtime_v3.md`；原科学协议保持登记hash。

## 专家参数的实际作用范围

- `dino_text_threshold` 在当前 Transformers 实现中只生成候选的文本标签；
  runner 保存标签但不使用标签筛选 anchor，所以它当前不改变伪目标。
- `dino_box_threshold` 筛选框；只要最高分有效候选仍保留，调阈值不改变选中框。
  跨过候选分数时可能让该 query 失去空间支持。
- TVG segment voting 可改变时间区间；SAM mask threshold 可改变框边界或空mask。
  最终只有改变映射后的PTD目标token或支持集合，才会改变此固定输入下的更新。
- 专家分数用于选候选，没有直接作为梯度的可靠性权重。当前两支full-F梯度分别
  单位归一化，低专家分数不自动带来小更新。

这些是代码作用路径，不是敏感性实验排名。本轮固定专家证据，27配置只识别
这些证据条件下K/rho/T-S权重的影响，不能据此判定专家超参数是否重要。
