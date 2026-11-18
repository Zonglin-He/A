# R172 — Joint score attribution and C/D reference verification

In 25 failed cases with both correct proposal identities available, the largest score difference favored the winner through the subject unary in 13 cases, the D-reference unary in 11 cases, and the left/right factor in one. Deleting one visible seed never reversed those margins. The analysis reused the sealed R171 candidate banks, 60 previously verified directions, and 30 physical frames. [Full report in Chinese](REPORT_ZH.md)

Restricting the reference to its verified identity on the 35 eligible directions increased Acc@50 from 5 to 7 and correct pairs from 3 to 12, with three subject IoU gains and two losses. All 46 nonempty D-reference banks made the original null unreachable. These are offline mechanism checks.

C/D linkage used mutual unique maximum overlap with the inherited shared point thresholds. All 129 links with independently identifiable physical ownership on both sides agreed on owner, out of 674 total links. C reciprocal rank improved new-frame reference Top-1 from 7 to 8 out of 32. On the R170 replay it remained 7 out of 28, versus 8 with the score-permuted control. The predeclared identity criterion failed. The optional independent and joint localization readouts were therefore not run. R158 with PCA remains the deployed readout.

[Pair attribution](results/pair_attribution.csv) · [Oracle](results/oracle_summary.csv) · [Identity discrimination](results/identity_summary.csv) · [Independent audit](SCALAR_AUDIT.json)
