# 重建 28 条源域案例

在已有三份单源 checkpoint、私有源域 RGB／点云／标定／标注，以及前序已封存源域预测的研究工作区运行。[父实验的生成和封存方式](../3eed_v1_native_seed_code_decoding_20261120/RUN.md) · [本轮清单及哈希](PROTOCOL.json)

```bash
study=studies/3eed_v1_k16_gt_error_attribution_20261120
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
crossplatform/env/bin/python "$study/cases.py"
crossplatform/env/bin/python "$study/render_cases.py"
crossplatform/env/bin/python "$study/summarize.py"
crossplatform/env/bin/python "$study/verify.py"
```

- `cases.py` 从父实验固定的 192 行中找出全部 18 条冻结漏选和 10 条 seed 误改。候选、seed logits、原赢家和参照对照沿用已封存文件。为拆开候选重叠投票前后的语言分，首次运行仅对这 28 行重放冻结前端。后续使用与预测摘要对应的逐帧缓存。原四槽由真实的前端取槽程序重建。输出 `results/paired_scores.csv`。
- `render_cases.py` 把整幅原图、未标记局部图、三种点云投影和标注汇集到 `review_PRIVATE/plates/`。可识别的原始数据仅保存在本地。
- `results/identity_review.csv` 是初次逐例读图后记录的身份及证据。复核者可据原图与点云逐行检查。单独的图像分类模型或二次人工投票均未参与。
- `summarize.py` 连接身份判断、两种读出的分差和 IoU，生成公开的 `CASES_ZH.md`、分组统计，以及本地带原图和归因的 `review_PRIVATE/index.html`。
- `verify.py` 从封存预测和源域标注独立重算两组清单、全 K16 的合格与失败排序、完整正负 Focal 代价及冻结分解。在全部 448 个混合框上比较两套有向框 IoU 算法，并用 24 种显式置换复核分量贡献。最后核验同帧证据页、锁定的 v1 核心文件哈希和公开表。

仅有公开制品时，可用 `results/identity_review.csv` 与 `results/paired_scores.csv` 按 `case_id` 一对一连接重算计数与模型分差。真实图册需要以上私有数据。两组均使用同源模型评估该源自己的标定帧。
