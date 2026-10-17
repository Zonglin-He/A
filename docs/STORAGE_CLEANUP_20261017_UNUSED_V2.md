# 2026-10-17 已完成的限定存储清理

经用户明确授权，删除2105个未选用下载包、非当前模型分片、下载残片和已闭合Teacher编码缓存，共103076231816逻辑bytes（95.997GiB）。磁盘可用空间36.02→132.01GiB，净增约95.99GiB。逐文件SHA/身份/mtime/大小、两次openFD/mmap及当前实验依赖pins通过。

| 类别 | 文件数 | 逻辑GiB |
|---|---:|---:|
|unused_dataset_packages|45|61.619|
|inactive_model_weights|4|15.445|
|abandoned_download|8|1.578|
|rebuildable_teacher_cache|2048|17.355|

保留当前R1/DINO/TA-STVG/TubeDETR权重、HC/Vid原始视频、Source前缀、context像素输入、Teacher原响应、预测、资格、失败记录、报告与负结果。VideoChat来源/config/tokenizer及分片SHA收据保留；下载集已提取的clips/frames/元数据保留。删除的模型/数据包需重新下载，退休Teacher缓存需重建，不宣称仍能立即逐tensor回放。每个缓存留单独退休marker，原研究结果不变；本次没有模型运行、GT读取或参数修改。
