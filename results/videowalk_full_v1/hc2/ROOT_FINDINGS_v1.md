# ROOT review: hc2 complete Source-feature VideoWalk extension

1902 queries; 227 parent clusters. All six main child stages and independent shards exited zero. Root read the complete report, absolute means, paired parent-cluster 10,000 intervals, short/long groups, distance support, signed tails, empty masks and measured cost. This review does not rescore GT or rerun any model.

|Spatial path (same Current interval)|vIoU|R.3|R.5|Native unobserved IoU|Conditional dense unobserved IoU|
|---|---:|---:|---:|---:|---:|
|PureMatched|25.177364|38.012618|13.932702|47.262804|49.641653|
|Matching|21.729104|31.913775|11.619348|40.027401|43.020957|
|Current|27.349044|41.219769|16.876972|52.942787|54.354908|

all:
- Current-Matching: delta vIoU +5.619940 pp; 95% parent-cluster CI [5.200882871598394, 6.049637513736688]; 1902 queries.
- Matching-PureMatched: delta vIoU -3.448261 pp; 95% parent-cluster CI [-3.904732778608287, -3.004144923525092]; 1902 queries.
- Current-PureMatched: delta vIoU +2.171679 pp; 95% parent-cluster CI [1.8399107531024494, 2.5196931008075563]; 1902 queries.

short_le5s:
- Current-Matching: delta vIoU +4.178797 pp; 95% parent-cluster CI [3.626398200434297, 4.7708371010825825]; 544 queries.
- Matching-PureMatched: delta vIoU -2.179244 pp; 95% parent-cluster CI [-2.89235476182095, -1.5313005500148464]; 544 queries.
- Current-PureMatched: delta vIoU +1.999552 pp; 95% parent-cluster CI [1.4691942488152219, 2.560668349356226]; 544 queries.

long_gt5s:
- Current-Matching: delta vIoU +6.197246 pp; 95% parent-cluster CI [5.688403684619416, 6.736888765253709]; 1358 queries.
- Matching-PureMatched: delta vIoU -3.956615 pp; 95% parent-cluster CI [-4.501861920614346, -3.434403159972668]; 1358 queries.
- Current-PureMatched: delta vIoU +2.240631 pp; 95% parent-cluster CI [1.8431414031224034, 2.6552260550309605]; 1358 queries.

Signed counts: {"Current-Matching": {"improved": 1424, "harmed": 287, "unchanged": 191, "decline_gt5pp": 28, "rescue_gt5pp": 718, "decline_ge10pp": 7}, "Matching-PureMatched": {"improved": 518, "harmed": 1190, "unchanged": 194, "decline_gt5pp": 593, "rescue_gt5pp": 170, "decline_ge10pp": 320}, "Current-PureMatched": {"improved": 1071, "harmed": 644, "unchanged": 187, "decline_gt5pp": 129, "rescue_gt5pp": 424, "decline_ge10pp": 33}}

Empty-mask accounting: {"native_unobserved_frames": 26081, "total_native_unobserved_frames": 108414, "queries_any_empty": 1401, "zero_box_no_fallback": true, "all_eight_original_expert_boxes_exact": true}

Interpretation: the result supports the comparison with this exact source-fixed Source-feature VideoWalk module adaptation. Its lower performance than pure interpolation is retained; it is not evidence against the full original learned-encoder VideoWalk method or every strong propagation algorithm. Same expert input and interval do not hard-lock Current observed outputs. Current retains its original soft E/expert selection, with its observed-frame losses recorded separately. No target retuning or method/anchor/paper changes.

Native unobserved IoU averages GT-supported native sampled unobserved frames per query. Conditional dense unobserved IoU restricts support to the common predicted/GT-event intersection, excluding exact eight observed physical frames; empty support is missing, not zero. Distance bins are 0-.5, .5-1, 1-2, 2-4, >=4 seconds. A zero-support bin is unscorable, not equivalence. The existing exposed target lists and pointwise CIs do not constitute an unseen confirmation or familywise guarantee.

All Source acquisition, matching, provenance IO, cold replay, startup and actual process costs are archived. Source/new-forward averages are separated from reused inputs. The first matching-pass CPU clock excludes acquisition/cold audit and is not a matched deployment comparison. Model load is counted once for its PID/start; nested clocks must not be added. Reused Source/expert acquisition costs are inherited, not zero.

Full signed review: ROOT_review.json. Full short/long, conditional distance, all positive/negative/null intervals: report.json. Whole suite still running until both domains and final root/publication verification complete.
