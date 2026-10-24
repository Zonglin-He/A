# 下一轮双主干：缓存保留、磁盘与执行顺序

2026-10-23用户要求尽可能缓存以加速下一轮，并明确TubeDETR也要运行；随后授权清理不用的磁盘内容。先完成当前LLaVA-ST两域全量的优先级继续有效。新方法固定为此前取回的main `a238705a`，TubeDETR纳入下一轮，不恢复历史调参队列。

## 已完成的缓存准备

独立工具位于 `tools/reuse_cache/`，不修改当前146项绑定源码或运行计划。

- `catalog.py` 将现有两主干、两域各128条完整空间捕获登记为512条可寻址引用，同时登记原生全量预测与内容绑定专家缓存。完整统计见[缓存准备回执](../results/next_backbone_cache/preparation.json)。旧文件不搬移、不复制、不重写；TubeDETR专家目录实际指向旧工程归档，这个链接及其目标必须保留，不能当作废弃归档清除。
- 索引核对完整捕获的seal、receipt及计划绑定；全部大型payload在真正复用前才检查SHA256。当前全量逐query receipt只表示该query已写完，不表示总体完整，不允许据此提前评分。
- `Catalog.find()`按主干、类型、域和原请求hash查找；`Catalog.verified_path()`在读取前核验receipt与payload哈希。`Catalog.teacher()`按RGB字节、八帧形状、文本和完整专家指纹做cache-only读取，缺失就报错，不偷偷新启动专家。两个旧管线的teacher fingerprint不同，不能仅因都叫ReferDINO就混用。
- `ArrayStore`按模型、代码、输入、帧采样、表示、文本和seed绑定身份，以无损压缩NPZ保存数值；保持dtype和逐值精度、拒绝同身份不同内容、使用原子提交和写锁、不使用pickle、不自动淘汰旧证据。默认新增缓存上限64GiB，同时保留至少25GiB文件系统空闲。调用方必须检查`not_cached_capacity`；必须保存的证据用`required=True`，容量不足直接停止该采集，不能悄悄丢弃。
- 视频级跨query缓存必须提供真实冷/缓存一致性的独立性验证指纹。相同视频路径不能证明采样RGB、clip范围、时间编码和预处理相同；未经验证仍按query保存。

四条已有捕获（两主干×两域）完成payload hash核验及小型Source/原生区间/帧网格/八位置无损缓存逐位回读；已存在的三组完整预测与相应原生数组逐位一致。另核验两份专家payload的八框/八mask和有限值。LLaVA HC全量尚不存在，因此该组没有full对照。范围是抽样工程复查，未重扫全部30GB捕获或16GB模型，没有新GPU、专家或GT评分。

索引刷新命令：在项目根执行 `python -B -m tools.reuse_cache.catalog`，数据库位于 `runtime/reuse_cache/catalog.sqlite`。数据库和原始缓存保持私有，不加入Git。工具已可用；当前运行中的worker不会被热注入，新worker接线仍需独立协议和真实资格验证。

## 下一轮保存优先级

1. 所有query保留Source、原生区间、全部所需原生时间logits/候选先验、帧对应、八观察与内容指纹，不能只保留最后两个时间数字。
2. 专家八框、原生mask、selected query等按精确内容键保留，优先复用现有约1.9万份专家证据；缓存miss按新协议显式处理。缓存部署成本与冷启动完整成本分开记录。
3. 保存新版实际所需的pre-text局部网格、valid支持和空间对应；新SimplePath轨迹必须重新pool，旧BoxFixed pooled features不替代。优先保证两域原128完整网格，再按64GiB/25GiB空间限制扩展。经真实验证后，query-independent视频前缀可按内容去重，避免为同一视频多次存储大网格。
4. 可复用的候选几何、辅助路径、池化特征和CPU结果都绑定其算法配置hash。方法变化时仅失效相关派生层，不重算仍精确匹配的Source/专家层。

当前全量进程未保存的中间量已从内存释放，无法从预测框恢复。LLaVA完整网格按每query约116MB保存全部12205条会超过1TiB，不能在当前磁盘无限展开。现有128捕获和全部专家缓存保留；原生全量输出已有索引。新增数值缓存工具不等于当前全量已经补存这些缺失特征，也不等于新版适配器完成。

## 磁盘清理

只删除pip HTTP/下载wheel缓存，以及通过Conda `clean --tarballs --index-cache`清理安装包压缩件和索引。实测文件系统可用空间增加10,068,672,512字节（约9.38GiB）；清理后可用102,646,497,280字节（约95.6GiB），期间其他进程写入会影响瞬时值。

安装环境、CUDA扩展、模型、媒体、全部正负结果、封存文件、原生特征和专家缓存保持。未清理用途未确认的uv归档、其他项目模型/数据或其他任务临时目录。详细本地回执：`outputs/maintenance/disk_cleanup_20261023.json`。

## 后续执行

[下一轮请求记录](../results/next_backbone_cache/next_run_request.json)明确两主干及先后依赖，属于用户授权范围和待办记录，不冒充已冻结可执行计划。当前LLaVA Vid10303与HC1902全部封存、独立评分、核验发布后，再按新方法分别完成TubeDETR与LLaVA适配资格和固定方法评估。不得中途切换、不扩大旧网格；原128面板已曝光，相关结果不能称独立确认。TubeDETR原生末层时间先验和pre-text网格、LLaVA合法候选时间证据均须实际接通并验证，接口不通过时保留失败原因，不能伪造后验。

2026-10-24容量执行修订：TubeDETR新cache达64GiB保护后Root显式恢复，现有近64GiB缓存全部保留且停止追加；不影响逐query必须保存的最终native/五行预测证据。缺失前缀冷计算，精确命中仍无损校验，0新专家，64GiB上限/25GiB下限不放宽。适用修订和入口见docs/simplepath_tubedetr.md及results/simplepath_tubedetr/capacity_recovery_plan.json；原缓存工具本身、旧证据及固定数学保持。后续LLaVA缓存容量也须按剩余空间准备，不能删除本次缓存或静默扩容。
