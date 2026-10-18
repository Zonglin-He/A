# Measured interpretation of the endpoint-target control

HC: endpoint changes complete vIoU from26.861045 to26.895387%, versus26.830781%
for the original residual target. Endpoint-E is+0.034341pp
[-0.002470,+0.089014]; Endpoint-Residual is+0.064606pp
[+0.004755,+0.143690]. Observed actual-node sIoU improves+0.289037pp over E
[+0.062791,+0.562924], versus the old -0.291001pp. This supports retaining
the endpoint target as a candidate rather than extending the original residual
rule. It does not establish an independent or stable complete-tube gain over E.

Vid: Endpoint-E complete vIoU is+0.098649pp [+0.028249,+0.185633]; its
+0.001464pp difference from Residual has an interval crossing zero
[-0.027695,+0.027134]. Observed sIoU is+0.810981pp versus E, with an interval
crossing zero; its -0.134327pp difference from Residual is also uncertain.
The endpoint target is not uniformly superior across endpoints or examples.

All16 HC queries remain: Endpoint-E has8positive/2negative/6neutral, versus
Residual3positive/7negative/6neutral. Vid remains8positive/1negative/7neutral.
No complete-vIoU loss exceeds5pp. H022/H031/H032/H045/H061 change from
negative to positive complete deltas against E. H043 changes from+0.016504pp
under Residual to-0.092265pp under Endpoint; preserve that negative case.

H032 contributes+0.024206pp to HC's+0.034341pp mean endpoint gain. The
remaining15 still average a small positive+0.010811pp as a posthoc sensitivity
calculation. All main means and CIs retain16; do not claim all improvement is
one query or present the remaining15 as a new score. Exact values are in
SENSITIVITY.json.

HC mean observed image-center/log-size displacement changes from
0.002837157/0.014675791 to0.001955777/0.009647295. Vid changes from
0.004007056/0.020298388 to0.004072712/0.020381742. Thus the target change
also changes actual amplitude; this experiment is not a matched-amplitude
proof of direction-only causality.

Unobserved actual nodes remain bit-exact E. HC's+0.034341pp comprises
+0.005562pp at observed nodes and+0.028780pp through benchmark interpolation;
Vid's+0.098649 comprises+0.013351/+0.085298pp. These are metric
contributions, not new visual evidence. All tIoUs are identical.

Freeze only the candidate endpoint rule specified in FROZEN_CANDIDATE.json.
Current deployment is unchanged. No new cohort, center query,24-state teacher
rerun or dynamic OPD is started. This finite comparison changes the estimation
target and uses old24 feedback; it cannot establish that24 conditional states
are better than a center-reference call, nor an on-policy benefit.
