# 项目和历史分支索引

从 `main` 查看当前 Qwen 方法和已发布结果。其他研究线各自保留原有入口、结果和历史版本。

| 研究线 | 常用入口 |
| --- | --- |
| **Qwen 单帧三维指代** | [方法与普通 Validation 主表](qwen-grounding.md) · [双队列论文实验](../studies/3eed_qwen_height_only_paper_20261123/REPORT_ZH.md) · [几何消融与复放](../studies/3eed_qwen_extent_height_only_20261122/REPORT_ZH.md) |
| **外部复测** | [同高度对照、WildDet3D 和 Talk2Car–mini](../studies/3eed_height_only_transfer_20261123/REPORT_ZH.md) · [Talk2Car 调参](../studies/3eed_talk2car_parameter_tuning_20261123/REPORT_ZH.md) |
| **旧版 3EED** | [四尺度联合读出](../v1/RUN.md) · [旧版方法登记](../v1/crossplatform/CURRENT_METHOD.json) · [跨平台实验](../studies/3eed_four_scale_direct_subject_20261120/REPORT_ZH.md) · [候选排序复测](../studies/3eed_v2_ordinary_val_subject_ranking_20261119/REPORT_ZH.md) |
| **HiSpatial** | [观测能力审计](../studies/3eed_hispatial_observation_audit_20261121/REPORT_ZH.md) · [Qwen 96 帧对照](../studies/3eed_hispatial_qwen96_confirmation_20261121/REPORT_ZH.md) |
| **STVG 时空视频指代** | [方法](method.md) · [研究历史](history.md) · [原归档](archive.md) · [旧登记](../CURRENT_METHOD.json) · [共享 ROI 登记](../CURRENT_METHOD_v2.json) |
| **AVIS、GREAT 与多方向 TTA** | [首轮报告](../studies/three_direction_tta_20261029/research/REPORT.md) · [第二轮报告](../studies/three_direction_tta_round2_20261030/REPORT.md) |

也可以展开 [全部实验目录](studies.md)，按主要研究线找报告和相关文件。

早期 v1 的两个版本保存在各自的发布提交中。 [组件消融](https://github.com/Zonglin-He/A/blob/dd5b3c94393420a11de6f207daf2a506f973f9cb/v1/ablation/README.md) · [前端简化](https://github.com/Zonglin-He/A/blob/d0592f91d698a632d5c40783c3ef9518625411aa/v1/verification/README.md)

## 历史 GitHub 分支

`main` 是默认入口。下列固定提交记录本次整理开始时各分支所在位置。
旧首页和整理前的文件可从 [main 原提交 `77ea1436`](https://github.com/Zonglin-He/A/tree/77ea1436c9c1e792ac99acab4530c08e34a83680) 查看。


<details>
<summary>3EED 研究分支（10）</summary>

| 分支 | 固定提交 |
| --- | --- |
| `research/3eed-full-pool-verification-20261119` | [`612da5e3`](https://github.com/Zonglin-He/A/tree/612da5e34e4eda1c6d5ac7df0d8d04345f83998a) |
| `research/3eed-r174-strong-anchor-20261119` | [`261676dd`](https://github.com/Zonglin-He/A/tree/261676dd7ed9cc9b6d87aa9514d5a083b4e5c860) |
| `research/3eed-r175-relational-support-20261119` | [`b3a155fe`](https://github.com/Zonglin-He/A/tree/b3a155fe8bb3eddfb0f311de7763e4bd22acf407) |
| `research/3eed-r176-dual-role-coverage-20261119` | [`e3970edf`](https://github.com/Zonglin-He/A/tree/e3970edf9427d9bfc7aeaca2b8dfa6f2a197d1f1) |
| `research/3eed-r177-r174-fine-ablation-20261119` | [`1b4dc50a`](https://github.com/Zonglin-He/A/tree/1b4dc50a318332a1245aced6efba3d2e017af32c) |
| `research/3eed-r178-r174-combined-simplification-20261119` | [`b8cc5449`](https://github.com/Zonglin-He/A/tree/b8cc5449d3cbd709a71fdc64c43edf00af28f561) |
| `research/3eed-recurrent-role-support-20261119` | [`2699fdcb`](https://github.com/Zonglin-He/A/tree/2699fdcb9b246f1764a0bb84d383cd35f3cefe82) |
| `research/3eed-role-geometry-factorial-20261119` | [`ea26e7a4`](https://github.com/Zonglin-He/A/tree/ea26e7a4e0cc2c0b61f660f62042bf2bf07b6376) |
| `research/3eed-source-reference-20261119` | [`6da7432b`](https://github.com/Zonglin-He/A/tree/6da7432bb2518e0e3688e57ca6fd21bbd479043f) |
| `research/3eed-verified-competitors-20261119` | [`da78a7e9`](https://github.com/Zonglin-He/A/tree/da78a7e9dcb5ea8ba2de497e60f5f3944d2119ca) |

</details>

<details>
<summary>HiSpatial 和其他专题（3）</summary>

| 分支 | 固定提交 |
| --- | --- |
| `research/codex-overnight-20261119` | [`92e2a177`](https://github.com/Zonglin-He/A/tree/92e2a177e83326a9b14c461ccb4a50fb6814170f) |
| `research/decota-paper-baseline-scoring-v1` | [`f15641a0`](https://github.com/Zonglin-He/A/tree/f15641a072086619efd95d05ead3c539d37f52c5) |
| `research/hispatial-zero-shot-spatial-reasoning` | [`4c5ffbcf`](https://github.com/Zonglin-He/A/tree/4c5ffbcf5e74a4a9328b67c2ecb05cb5de6633f4) |

</details>

<details>
<summary>STVG、Decota 与 TubeDETR（6）</summary>

| 分支 | 固定提交 |
| --- | --- |
| `research/decota-spatial-opd-tuning-v1` | [`736ae68e`](https://github.com/Zonglin-He/A/tree/736ae68e9cd0e76080e79fe0c3180632bcdd5bf5) |
| `research/decota-spatial-opd-v1` | [`cf9b6d8e`](https://github.com/Zonglin-He/A/tree/cf9b6d8e94af5f09dfe70f46061b599ae981f5d7) |
| `research/stvg-opd-hc2-coordinate-v1` | [`320c6294`](https://github.com/Zonglin-He/A/tree/320c6294a48c6200045be52362c67e410cbf0eeb) |
| `research/stvg-opd-paper-hc2-revision-v2` | [`073def4b`](https://github.com/Zonglin-He/A/tree/073def4b4b08f076aa603531cf4a61605eebec23) |
| `research/stvg-opd-paper-v1` | [`30dd1d53`](https://github.com/Zonglin-He/A/tree/30dd1d5325cb015932b8ed68540b908a2033bc04) |
| `tubedetr` | [`1f114ad2`](https://github.com/Zonglin-He/A/tree/1f114ad2da9abad7cebfc8cae6faca76e799d653) |

</details>

<details>
<summary>早期 STVG 快照（6）</summary>

| 分支 | 固定提交 |
| --- | --- |
| `archive/research-pre-general-20261023` | [`fa2805ce`](https://github.com/Zonglin-He/A/tree/fa2805cecef8fa2a929aea1f5755d3102b33d9b0) |
| `archive/research-pre-k16-20261023` | [`b9345a8a`](https://github.com/Zonglin-He/A/tree/b9345a8a753baa969dd8249e22b7b675488dbd10) |
| `archive/research-pre-mainline-20261022` | [`c310af58`](https://github.com/Zonglin-He/A/tree/c310af5888dca78d6ea7b0a280cb67d209a90337) |
| `archive/research-pre-native-prior-20261023` | [`e35aca3e`](https://github.com/Zonglin-He/A/tree/e35aca3e2abdcfe743c4042f903f30df8efc1002) |
| `archive/research-pre-native2d-20261023` | [`601b9ccf`](https://github.com/Zonglin-He/A/tree/601b9ccf7c7e6ec629dd92b0a0db6c6c82e3ccac) |
| `archive/research-pre-simplepath-20261023` | [`0c83e30a`](https://github.com/Zonglin-He/A/tree/0c83e30ab83a51f99389baee8741d69db56f50ef) |

</details>
