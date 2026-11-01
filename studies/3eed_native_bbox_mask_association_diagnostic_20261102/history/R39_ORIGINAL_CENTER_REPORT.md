# R39：同一冻结 Qwen 的 native bbox 输出接口

## 裁决

本轮已完成真实 A208 推理、预测封存、CPU 逐条重放和任务评分。固定 native bbox 接口的两平台平均 mIoU 为 **20.7337%**，相对冻结 source 提高 **1.0701 pp**，但相对 R38 兼容 point 下降 **2.0066 pp**，相对旧 G 下降 **3.9937 pp**。它没有通过预登记的总体强对照条件，**不接参数 TTA，不替代当前精度默认方案**。这只裁决当前 3B、固定单图和框中心读出配置；不能推断所有 bbox grounding 或全部外部 VLM 监督无效。

所有 A208 都是已经暴露的开发数据，104 条 Drone、104 条 Quad，不能称为新的独立确认。FINAL、T、U 未读取，没有源模型训练、学生参数更新、三维框生成或修正。

## 最小变化与官方依据

同一冻结 Qwen2.5-VL-3B、bf16/sdpa、完整原描述、单张未标记 RGB、原最小/最大像素设置、贪心解码和 96 token 上限保持不变。只将 native point 的输出模板换成一个 `bbox_2d` xyxy 框或 null。原 point/SoM、R29 和旧 G 的 11 个缓存对照均未重新生成，CPU 审计确认它们全部 query 和原 source 三维框逐位不变。

官方固定历史 notebook 的 REC 示例支持 native bbox JSON 接口；本地原文摘录保存在 `OFFICIAL_EXCERPTS_SNAPSHOT.json` 和 `OFFICIAL_REVIEW_SNAPSHOT.md`。该示例引用 7B，不能写成本地 3B 的 RefCOCO 协议复现，也不能根据两个不同 benchmark 的分数预先认定 bbox 比 point 强。参考：[官方固定提交定位示例](https://github.com/QwenLM/Qwen3-VL/blob/f61c3cdcffac3aa3dba0bde01c9763b87db0dccb/cookbooks/spatial_understanding.ipynb)、[Qwen2.5-VL 技术报告](https://arxiv.org/html/2502.13923v1)。

安全解析从第一条调用前固定为严格 JSON：根对象或单元素对象列表，唯一 `bbox_2d`/`bbox2d`/`bbox` 定位键，可附加一个只保存不使用的字符串 label。多框、多定位键、重复键、未知字段、布尔或字符串坐标、非有限值、坏 JSON 均拒绝。框必须严格正面积，xyxy 各轴位于声明的实际输入范围；不交换边界、不剪裁、不切换到 0–1000 单位、不循环重问。文字和图像文字始终是不可信场景数据，模型输出不被执行。

坐标合同仍为固定 Qwen2.5 的实际 processor `image_grid_thw × 14` 输入像素，逐轴反变换到原图。二维框中心只用来查询原独立 mask：恰好命中一个 mask 后，查原 R37 完整 mask→3D 唯一组，返回组内固定原 XYZ source 分数最高的 query。原 R29 AB-only 字典作为 oldsubset 读出消融；两个臂的 VLM 输入完全相同。框中心不是三维物理中心，既不拟合边界，也不引入框 IoU 门、最近 mask 或手工加分。

## 全体指标

mIoU，单位 %。括号中的变化均相对同平台 frozen source。

| 方法 | Drone | Quad | 两平台平均 |
|---|---:|---:|---:|
| Frozen source | 12.7430 | 26.5841 | 19.6636 |
| R29 direct | 13.6502 | 27.0048 | 20.3275 |
| 旧 G | 15.9786 | 33.4761 | 24.7274 |
| R38 compatible point，完整映射 | 15.4144 | 30.0663 | 22.7403 |
| R38 compatible SoM，完整映射 | 9.5595 | 27.2884 | 18.4239 |
| **R39 bbox，完整映射** | **12.2321 (−0.5109)** | **29.2353 (+2.6512)** | **20.7337 (+1.0701)** |
| R39 bbox，旧 subset | 12.1397 | 29.1140 | 20.6269 |

| R39 完整映射 | Drone | Quad |
|---|---:|---:|
| Acc@25 | 25.0000 | 51.9231 |
| Acc@50 | 6.7308 | 31.7308 |
| 相对 source 改善／恶化 | 18／13 | 23／16 |
| 改变 query | 46 | 50 |
| 合法映射／source fallback | 76／28 | 88／16 |

完整映射相对旧 subset 只提高平均 **0.1069 pp**。bbox 在两个平台都低于 compatible point，因此不能把当前不足全部解释成旧候选映射覆盖损失。固定场景聚类的描述性 bootstrap 区间、逐场景表、全部 13 臂的 mIoU/Acc@25/50 和所有负例分别存于 `FACTOR_CONTRASTS.json`、`SCENE_ROWS.json`、`TASK_SUMMARY.json`、`ALL_NEGATIVES.json`；少量已暴露场景的区间不是独立确认或大样本证明。

## 格式、映射和剩余解释

208 条正式输出中，202 条通过结构和框几何检查，4 条坏 JSON、2 条框越出声明输入范围。没有 null。拒绝项照原规则 UNKNOWN/source fallback，未按任务结果修复。

完整映射的失败分解：Drone 有 17 条目标 mask 无三维组、6 条中心落到重叠多 mask、3 条无 mask、2 条框越界；Quad 有 6 条无三维组、6 条重叠多 mask、4 条坏 JSON。这些不是一个统一的“VLM 语义错误”标签。正确二维框的中心也可能落在空隙、遮挡物或邻物；当前实验只评价了固定中心→mask 接口，不能用任务 IoU 单独认证二维主体身份。

同一预先封存 24 条身份注释的 bbox 对照由独立代理追加，注释不改、不按预测选样、不用来筛监督；它是有历史文件暴露声明的 assistant provisional review，不是专家 gold。待独立 join 结果另行补充，不能为保留该接口而重标注。

R37 的两次模型定位诊断（原 1/2、后续真实方块 1/8）及单位解释界限全部保留。它们不要求 bbox 额外达到完美命中，也没有用于拟合坐标校准。本轮没有新增合成模型扫测。

## 测量与成本

原 `PLAN_LOCK.json` 在首次调用前固定。CPU schema、注入、边界、逆变换、唯一 mask 和原 system 逐字一致性检查通过。复用原 12 条真实预检：11 条合法框/null，9 条合法映射，6 条改变 query；该门检查可执行链条而非准确率。12 条生成直接复用到正式 208 条，正式完整阶段只新增 196 条。

首次 attempt001 在第一个实际模型生成完成后，由于遗漏 numpy 导入，计算图像散列时 `NameError`，raw 未保存。原代码、ledger、失败目录和日志全部保留。Root 在任何任务评分前授权纯技术 attempt002：补导入，并在解码后原子保存 raw 和标量元数据，再进行解析/图像散列；后续失败可读取已存 raw，避免重新生成。模型、prompt、schema、坐标和映射未变。没有旧答案可供选择，只有这一条同输入技术重跑。总实际上限明确登记为 **209 = 1 次 raw 丢失实现失败 + 208 次正式成功**。失败次 token 数和耗时未知，未伪造或从调用次数中抹去。完整技术记录见 `RUN_TECHNICAL_HISTORY.md`。

正式成功 208 次生成的输入 token 为 254003，图像 token 为 208000，输出 token 为 6155，模型生成时间约 143.996 秒；这是成功正式调用的生成计时，不含加载、CPU 审计或那次丢失 raw 的未知成本。实际调用总数仍报 209。原 11 个缓存臂仅作为历史同数据参照，不能把其成本写成这次新产生或宣称所有接口等成本。

208 条正式 raw、原子 stage、完整 13 臂预测和原框向量先封存，随后 CPU 重放全部通过：原 11 臂不变、坐标和解析一致、中心与唯一 mask 映射一致、源三维框精确相同。封存 SHA：`188b8199e6f152e0451b64505cf28b23e56784864a0c5561e570b15a300dd783`。CPU 审计通过后才读取任务 GT 评分。

## 证据入口与后续边界

- `PLAN_LOCK.json`、`TECHNICAL_FIX_BUDGET_AUTHORIZATION.json`、`ATTEMPT002_IMPLEMENTATION_LOCK.json`：原方案及技术修复预算。
- `CPU_TECH_CHECKS.json`、`CPU_AUDIT.json`：软件合同和封后完整重放。
- `TASK_SUMMARY.json`、`FACTOR_CONTRASTS.json`、`COVERAGE_UNKNOWN.json`、`SCENE_ROWS.json`：总体与分层结果。
- `ALL_ROWS_PRIVATE.json`、`ALL_NEGATIVES.json`：逐条结果及所有负例；公共导出应脱敏，不能上传原图、原描述、权重或私有缓存。
- `COSTS.json`、`DECISION.json`、`RUN_TECHNICAL_HISTORY.md`：成本、未采纳裁决与失败保留。

停止当前固定 bbox 配置的 TTA 接入。bbox 四坐标输出是官方支持的接口，但当前结果没有补上可靠实例偏好这一前提；在监督不可靠时增加学生更新层或拟合步数不能证明能解决问题。后续若另有新机制，必须另行登记其信息增量和同信息无更新对照，而不能基于本轮少数成功例继续改门。
