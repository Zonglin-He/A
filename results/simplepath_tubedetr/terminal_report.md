# TubeDETR SimplePath/Native2D 完整双域终态

Vid10303/732与HC1902/227均完整seal、独立CPU评分，Root已首核全部12205预测/receipt/request/plan/有限数组及评分均值；最终核验沿用Vid首核小hash而未重扫Vid/16GB模型。全部原生权重冻结，新增专家0，标签只在完整封存后离线读取。

|域|方法|m_vIoU|dense sIoU|tIoU|
|---|---|---:|---:|---:|
|vidstg|Source|13.116927|41.437894|28.674021|
|vidstg|Spatial|12.364156|36.794431|28.674021|
|vidstg|NativeMBR|16.338419|41.437894|36.610870|
|vidstg|Temporal|15.825489|41.437894|35.231971|
|vidstg|Full|14.916728|36.716875|35.231971|

vidstg Full−Source配对差(pp)：v +1.799801, CI95[1.336119,2.272636], s -4.721019, CI95[-5.391734,-4.040540], t +6.557950, CI95[5.803885,7.370939], unobserved -4.799570, CI95[-5.461798,-4.120723]。v下降>5pp 1875条，时间或v下降>5pp并集2974条。
|hc2|Source|15.425087|55.894585|26.800739|
|hc2|Spatial|14.916925|52.339202|26.800739|
|hc2|NativeMBR|25.823591|55.894585|45.643423|
|hc2|Temporal|26.722620|55.894585|47.322454|
|hc2|Full|25.187525|52.256528|47.322454|

hc2 Full−Source配对差(pp)：v +9.762438, CI95[8.841635,10.655797], s -3.638058, CI95[-4.526865,-2.768008], t +20.521715, CI95[19.236163,21.769096], unobserved -3.706662, CI95[-4.596498,-2.835143]。v下降>5pp 300条，时间或v下降>5pp并集368条。

Source为原MAP时间及原框，Spatial只换新空间路径，NativeMBR仅对原生末层先验做物理MBR并保留原框，Temporal加入单组Native2D反馈但保留原框，Full同时新空间路径和新时间。五行控制及全部逐query/父ID哈希匿名标量保留于各域all_signed_full_results.json.gz，不只保存有利样例。

所有结果使用冻结main a238705a数值方法和原两域输入；kappa64/beta1/单轮/context1.5、gamma Vid1/HC.5、SimplePath权重.5/.5/.25，未根据开发或full成绩调参。原开发面板及此前研究已曝光、且包含在full内，不称严格独立未曝光测试；父bootstrap描述性区间未做研究曝光校正。

缓存达到64GiB后的容量停止已显式归档；修订只停止新可复用缓存追加，原绑定推理/参数/模型/已有8591输出不改，缺失前缀冷算，原生logits/prior和全部预测仍保存。容量修订被域及总final引用，cache加速时延不能作冷部署成本。旧failure.json是已处理历史，当前无新错误。

原full/domain/total seals保留publication_pending原字节；最终发布状态结合outputs/maintenance/simplepath_tubedetr_final_publication.json晚于终态的回执。TubeDETR终态后按用户授权继续新版LLaVA真实原生条件时间先验/局部表示资格及固定方法评价；两域完整结果不等于该后续任务已完成。
