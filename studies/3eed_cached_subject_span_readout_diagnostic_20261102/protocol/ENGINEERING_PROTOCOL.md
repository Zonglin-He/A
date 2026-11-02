# R48 P0：冻结 source 的文字跨度读出诊断

本轮只使用已暴露的原 R24 D208、V64、M64 开发缓存，共 336 条；三个 split、平台及场景均应分报。它是在最终确认之后提出的诊断，不能当作新独立确认，也不修改、重评或反向选择已关闭的最终实验。运行边界禁止访问该最终池任何清单、预测、输入或标注文件。

固定的是 source query、token embedding、原始候选框和参数。所有新读出共享一次 float64 `softmax(query @ tokens.T / 0.07)`，保留所有实际 token、BOS／EOS 和 sentinel 的分母。没有重新运行 grounding encoder，没有新增 G、RGB、几何修框、TTA 损失、拟合或 GPU 工作。

## 五个固定输出及解释

1. **current_proxy_CPU64**：复现原先 first-lexicon-noun，找不到时采用原 all-content fallback；实际布尔 mask 必须与缓存完全一致。
2. **official_LITERAL_indices_DIAGNOSTIC**：复现原训练 snapshot 的 class→平台同义词优先级→第一个匹配及有限 char-to-token 回退，将原注释格式产生的 token 索引字面应用于当前缓存概率。它使用类别信息，只是诊断，不是可部署的 GT-free 方法或 oracle。
3. **text_only_subject_parser**：只从 caption 和冻结依存解析选主体 head，未知时回退第一个臂。
4. **official_charspan_REMAP_DIAGNOSTIC**：使用相同注释匹配字符跨度，固定扣除官方前导空格的一字符偏移后，映射到实际 tokenizer offsets；它仍使用类别信息，不按任务结果选择是否采用。
5. **cached_native_current_BRIDGE**：保留原来存下的 float32 source 分数及赢家，单独展示 CPU64 重放、并列和 argmax 差异；不静默替换历史 source。

官方格式为首尾空格包裹的 caption，实际缓存输入为原 caption 加 ` . not mentioned`。caption 的逗号／空格规范化已在原输入中完成。本轮不改变或重新格式化实际输入，也不对原 embedding 做第二次归一化。官方 literal 与 char-span-remap 同时保留，不能依据任务 IoU 选择一个表现更好的解释。

重建当前 tokenizer offsets 的长度和原代理布尔 mask 有精确桥接；原缓存没有保存历史 input IDs，因而不能声称已把历史 token IDs 逐位重放验证。能直接保证的是使用未改的缓存概率表示、既定实际文字构造和一致的布尔 mask。

## Caption-only parser

原 textZ 虽调用神经解析器，但 freeform 主体选择实际是第一句中第一个非 STOP 的 NOUN／PROPN，而不是矩阵句法主体。其已保存的 category／subject span 不能冒充完整依存 head 缓存。新 parser 接口不接收 GT 类别、框、图像、点云、官方选择跨度或任务分数。

规则优先级固定：显式 Object type 字段中的唯一名词 head；否则矩阵 nsubj／名词性 csubj；否则裸名词或 deictic-copular 的名词 root；再是具有唯一直接宾语的预定指令动词。协调、多个主体／宾语、非名词性从句主体、首个实质句没有明确角色等返回 UNKNOWN，而不是继续挑一个较早类别词或后面的参照物。

主 mask 只使用选中 **head 词的原始字符跨度**及其全部 BPE。compound／flat／fixed 后代只记录诊断，不加入主 mask，不把属性或整个 NP 同时叠入分数。所有名词节点必须具有与 raw caption 精确对应的 offsets；缺失时先 UNKNOWN，防止删掉真实 head 后把其 compound 子节点错误晋升为目标。

解析使用已下载且冻结的 Stanza CPU 模型和原处理器组合，没有模型下载、grounder／VLM 或 GPU。该阶段只读 caption，336 条解析及依存词字典必须先全体封存，再允许官方类别读取。UNKNOWN 保留，回退只使用原 current-proxy CPU64；不按类别或 IoU 修复 parser。

## 类别与 GT 框的读取隔离

官方诊断 helper 在 parser seal 后才导入／执行。旧索引 join 明确保留身份、路径、对象序号和容器哈希白名单；元数据通过 streaming allowlist 只构造目标 class 和 caption，并核对原 caption。该阶段不保留 GT bbox。

每个 official 臂独立处理自己的空 mask：不能因为 literal 为空就同时让 remap 回退，或相反。全部匹配／空值／偏移状态、正 token 数、完整分数及预测均封存。GT 框任务评分另需 root 授权，且只能在五臂全体预测封存和独立算术／schema 审计之后统一执行。

## 工程检查与证据边界

纯合成测试覆盖：主体出现在参照词之后、deictic copular、imperative object、协调主体 UNKNOWN、explicit head-only／compound 诊断，以及缺失 head offset。初版最后一项发现 schema 错误：过滤掉缺失 offset 的真实 head 后，compound 会错误晋升。该失败和未执行原锁已保留；root 在任何实际 NLP／类别读取前登记最小安全修复，正常语法规则未变。同一六项测试随后全部通过。

上述合成通过只验证实现合同，不证明真实主体判断准确。预先选取的盲文字核验仅为独立辅助身份／角色审查，不是专家金标，也不用于筛掉负例或选择规则。总体结果必须保留全部 UNKNOWN、fallback、失败、改善和恶化，按全部 336 条及各 split／平台／场景报告；不能从已暴露开发结果提出未见场景或 label-free 开发历史的主张。

本文件是预评分工程与协议骨架，不包含任何任务结果或采纳判断。实际运行、审计、评分与结果报告由 root 的分阶段授权和独立记录补充。
