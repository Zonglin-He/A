# R168 Selective reference binding on a fixed candidate bank

R168 exactly replays the R167 independent and joint assignments on all 18,620 R165 cross-platform direction records. Candidate membership, subject scores, geometry, camera relations and PCA are fixed. The primary summaries weight the six transfer directions equally. The 6,864-record camera-horizontal subset uses the same paired inputs.

| Readout | mIoU | Acc@25 | Acc@50 |
|---|---:|---:|---:|
| Independent R158 with PCA | 14.496% | 27.340% | 10.161% |
| Original R167 joint | 13.785% | 26.009% | 8.864% |
| Left/right only | 14.511% | 27.372% | 10.063% |
| Overlap only | 13.632% | 25.797% | 8.700% |
| Each source-calibrated rejection arm | 14.496% | 27.340% | 10.161% |

The old null lost on all 10,904 nonempty roles, with a minimum archived advantage of 1.0 for the best physical assignment across subjects. The old joint readout produced 124 Acc@50 rescues and 363 harms compared with independent selection. The overlap-only ablation produced 113 rescues and 376 harms. The left/right-only ablation produced 57 rescues and 77 harms.

The new reference term is the native role-token log mass relative to all valid tokens. It retains the existing point-attention contribution and overlap-discounted transfer onto non-native candidates. The shared selective rule uses old unary margin and role enrichment, fitted on 120 source-own frames with annotation-derived identities and no target label data. Its clipped source-specific probability caps are 0.1344, 0.3421 and 0.1391. All are below the fixed 0.5 operating point. Old, new and permuted selective arms rejected every physical role, so the fallback recovered the independent endpoint exactly. A blind 12-role source audit confirmed six labels, contradicted three and left three unresolved. Recording-out Brier scores favored the constant-rate source priors.

Independent reference identity judgments cover 54 directional edges. The new standalone semantic readout increased confirmed references from 8 to 12. On the 750-edge annotation/proxy cohort, confirmed references moved from 108 to 102. In the 41 independently verified edges where the fixed bank contains the true reference under final box ownership, supplying that identity improves Acc@25 counts from 19 to 21 and Acc@50 counts from 14 to 16. Fixing the predicted reference instead gives 15 and 8. The oracle uses the highest original reference unary within the correct identity and only diagnoses the sealed relation rule.

The archived signed intervention covers 2,526 direction pairs. Original joint selects two compatible signs in 1,574 pairs and meets the shared known-reference and distinct known-subject conjunction in 38. Strict paired subject and reference correctness stays at zero on ten authored directional pairs and two verified archived ones. All unresolved archived panel attempts remain in the release.

**Decision.** Continue to use independent R158 with PCA. The tested selective option and the current normalized reference score have no established joint localization contribution. The diagnostic evidence supports further study of independently verifiable reference identity before another solver change. No source checkpoint updates, target parameter updates or FINAL access occurred.

[Full Chinese report](REPORT_ZH.md) · [Protocol](experiment/PROTOCOL.json) · [Scoring audit](EVALUATION_AUDIT.json) · [Release manifest](MANIFEST.json)
