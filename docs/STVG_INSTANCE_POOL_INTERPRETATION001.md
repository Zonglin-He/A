# Correction: nonSource coverage is not whole-pool object discovery

The E0/E1/E2 scores and all candidate scalar data stay unchanged. The earlier emphasis on nonSource coverage needs a narrower interpretation. Source is also an actual displayed candidate. Missing good nonSource boxes does not imply the complete pool lacks the correct object. A Source selection is forbidden as an OPD reference to avoid direct self-distillation, but that admission rule does not remove Source from semantic pool coverage.

| Domain | Exact known windows | Source IoU>=.5 | E0 full pool>=.5 | E1 full pool>=.5 | E0/E1 nonSource>=.5 |
|---|---:|---:|---:|---:|---:|
| HC |43|30|31|31|18/30|
| VidSTG |26|8|10|11|4/11|

The large nonSource increase mostly makes a Source-near hypothesis available as a nonSource pseudo-reference. Full-pool geometric opportunity improves much less: zero extra HC windows and one Vid window at this descriptive threshold. Full-pool maximum mean IoU changes .6414→.6579 in HC and .4351→.4723 in Vid. None of these overlaps is independent semantic identity annotation.

Among the10/11 actual selections of the new hypothesis, mean box IoU with Source is .7007/.7115; zero are exact Source copies. Thus it is pixel-conditioned new geometry, but remains strongly Source-conditioned and cannot reliably discover an outside referent. Only5 HC and4 Vid such selections have exact GT:1/4 HC better/worse than Source and2/2 Vid; the other selected windows remain unknown. These small subsets do not establish that new reference geometry is consistently more correct than Source.

The finite modification is therefore a Source-conditioned foreground/pseudo-reference test. Its small Student gains do not prove new instance discovery, semantic identity recovery or independent geometric truth. The original pool review did not uniquely identify missing-candidate generation versus R1 misselection: Source already provides some recognizable instances, and low geometric IoU does not establish a different identity. The saved E1-E0 result tests this one candidate modification only.

A Source-box reference control might distinguish pixel-derived geometry from merely admitting a Source-like reference, but no extra fit/control is launched: the current finite plan allows E0/E1/E2 and one frozen confirmation, not an expanding matrix. The pre-score advancement rule and64-source recipe remain unchanged. This correction is published immediately while confirmation continues. All old reports, negative examples and prior exports remain available; this note narrows interpretation rather than rewriting predictions.
