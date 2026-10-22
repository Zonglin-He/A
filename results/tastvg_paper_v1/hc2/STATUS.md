# HC2 paper panel completed; dual-domain package still running

All 1902 queries / 227 parents sealed before independent CPU scoring. Source and
Full exactly reproduce the earlier published full results. 15216 official/dense
cross-checks have maximum error 1.33e-15. Runtime48/provider434 sources unchanged.

| Method | m_vIoU (%) | m_tIoU (%) | Unobserved actual sIoU (%) |
|---|---:|---:|---:|
| Source | 24.9948 | 48.6473 | 51.0497 |
| Interpolation | 25.1065 | 48.6473 | 47.3369 |
| SharedInit | 25.8515 | 48.6473 | 51.9662 |
| OursS | 26.8147 | 48.6473 | 52.7883 |
| Full | 26.7597 | 48.5808 | 52.8379 |
| NoLocal | 26.2917 | 48.5808 | 51.9662 |
| Independent | 26.4818 | 48.5808 | 52.2520 |

OursS-Source vIoU is +1.8199pp (parent-cluster95% CI +1.3317 to +2.3275),
OursS-Interpolation +1.7081pp (+1.4616 to +1.9652). Full-NoLocal is +.4680pp
(+.1964 to +.7473); Full-Independent +.2779pp (+.2324 to +.3254).
Full-OursS is -.0550pp (CI -.2331 to +.1199): no HC temporal benefit established.
These results do not authorize retuning or replacing the frozen Full configuration.

Fixed support stage analysis uses the same actual GT-supported samples and observed
partition. Observed effective queries1880, unobserved1902; empty groups excluded.
Full signed values and severe declines are retained beside report.json. Historical
project OPD is separately labeled, not a third-party generic TTA baseline.

VidSTG remains running. Fair timing and private qualitative review follow both
sealed/scored domains. This stage record is not completion of the dual-domain package.
