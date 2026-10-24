# TubeDETR空间sIoU参数开发：两域原128完整结果

用户授权扩大空间数值参数搜索；Source保持原生MAP及原框，时间使用此前封存的Selected_interval。每域2048配置的完整128预测先封存，随后独立CPU评分。原128和此前全量研究已曝光，以下为搜索后的开发结果，不是独立确认。

| Dataset | Queries | Source sIoU | Frozen SimplePath sIoU | Selected sIoU | Selected−Source pp |
|---|---:|---:|---:|---:|---:|
| vidstg | 128 | 48.616419 | 43.851250 | 49.189528 | +0.573109 |
| hc2 | 128 | 57.596687 | 51.434574 | 58.243473 | +0.646786 |

## vidstg selected parameters

```json
{
  "family": "simplepath",
  "geometry": 5,
  "gamma": 0.0,
  "observed_weight": 0.5,
  "unobserved_weight": 0.5,
  "link_weight": 4.0,
  "guard": 1.0,
  "admission_margin": 0.0,
  "alpha_observed": 1.0,
  "alpha_unobserved": 1.0,
  "width_scale": 1.0,
  "height_scale": 1.0
}
```

逐query sIoU下降超过5pp：5条。保留全部配置与匿名逐query标量，不能只公开赢家。


## hc2 selected parameters

```json
{
  "family": "simplepath",
  "geometry": 1,
  "gamma": 0.0125,
  "observed_weight": 0.0,
  "unobserved_weight": 0.1,
  "link_weight": 1.0,
  "guard": 0.1,
  "admission_margin": 0.0,
  "alpha_observed": 1.0,
  "alpha_unobserved": 1.0,
  "width_scale": 1.0,
  "height_scale": 1.0
}
```

逐query sIoU下降超过5pp：6条。保留全部配置与匿名逐query标量，不能只公开赢家。

## 归因与下一步

两个赢家均width_scale=height_scale=1、alpha_observed=alpha_unobserved=1；不是单纯尺寸校准或直接混合回Source。Vid gamma0取消初始化全局偏移，affinity温度.06/topk20、连接4、保护1；HC gamma.0125、温度.1/topk5、观察权重0/未观察.1、连接1/保护.1。新增保护与候选margin属于明确记录的数值扩展，不冒充原固定上游协议。

两域参数已锁定。所选几何需要真实原生特征；独立全量控制器等待当前LLaVA全部GPU阶段完成后才取得共享锁。每域先对一条已有query进行Source/特征/所选输出逐位资格，再进行完整Vid10303、HC1902，完整seal后CPU评分，不用全量重新选参。尚无本轮全量成绩。

128工程检查加完整队列条件检查合计129CPU tests通过；这些检查不构成新的GPU资格或全量成绩。
