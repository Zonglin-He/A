# R12 无标签机制计数

作者侧实现解释；未读取任务标签、分数、图像、点云或特征。

all: N=146, anchorunique=97, realretrieved=291, attempted=194, PnPpass=116, staticpass=41, admittedanchors=23.

drone: N=50, anchorunique=22, realretrieved=100, attempted=44, PnPpass=10, staticpass=1, admittedanchors=1.

quad: N=96, anchorunique=75, realretrieved=191, attempted=150, PnPpass=106, staticpass=40, admittedanchors=22.

各臂接受/实际变化/尺寸/求解器、每层拒绝与逐点核验数据见SUMMARY.json。所有NA和零变化样本保持在原分母。

- No target GT, task scores, annotations/index sidecars, images, masks, raw points or source features decoded.
- Input-only frame index and pairmap contents are not reread; their existing hashes bind the observation selection.
- Retrieved, actually attempted, accepted pair and geometry-active counts have different denominators.
- Independent masks/class family/unique track/static support do not verify literal referring-expression identity.
- Clock is unverified; this is accepted overlapping-view evidence, not a certified continuous-video/velocity/online claim.
- Current PnP backend was registered after five input-development pairs with different-validation SIFT/ICP/PnP probes; counts do not prove algorithmic superiority.
- Legacy named multiview_maskmatch_only_direct is unfiltered samegate aggregation, not anchor-only.
- Main perpoint filtering changes loss support and normal context; samegate does not mean same accepted face set.
- Confidence, coverage, small residual or changed geometry is not task utility. No neural TTA/model update is executed.
