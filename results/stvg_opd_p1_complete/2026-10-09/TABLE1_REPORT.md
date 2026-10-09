# Table 1: fixed Spatial OPD and matching sealed baselines

Clean native cross-domain transfer on all official target queries and the original three orders. Parent macro averages queries and orders within each parent; official query macro averages each query over orders. The two target configurations were selected on exposed 32-parent development sets; those parents are separately excluded below. Remaining parents still have project history exposure.

| Target / method | Queries / parents | Official query m_vIoU % | vIoU > .3 % | vIoU > .5 % | Parent m_vIoU % [95% CI] | Parent m_vIoU % excluding 32 tuning parents |
|---|---:|---:|---:|---:|---|---:|
| hc2 / spatial_opd | 3482 / 237 | 22.583 | 32.970 | 11.784 | 23.510 [22.310, 24.777] | 23.152 |
| hc2 / source_only | 3482 / 237 | 20.017 | 27.570 | 7.639 | 21.072 [20.007, 22.204] | 20.743 |
| hc2 / tent_stvg | 3482 / 237 | 16.598 | 20.132 | 5.035 | 17.437 [16.559, 18.389] | 17.123 |
| hc2 / sar_stvg | 3482 / 237 | 19.939 | 27.465 | 7.486 | 20.994 [19.927, 22.122] | 20.668 |
| hc2 / dino_refine | 3482 / 237 | 20.673 | 28.920 | 9.075 | 21.888 [20.693, 23.176] | 21.377 |
| hc2 / target_trained_reference | 3482 / 237 | 29.056 | 45.951 | 14.302 | 29.895 [28.840, 30.966] | 29.691 |
| hc2 / eata_stvg | 3482 / 237 | 20.017 | 27.570 | 7.639 | 21.072 [20.008, 22.204] | 20.743 |
| vidstg / spatial_opd | 10303 / 732 | 14.097 | 17.904 | 7.037 | 15.376 [14.490, 16.284] | 15.464 |
| vidstg / source_only | 10303 / 732 | 11.129 | 12.831 | 3.300 | 12.065 [11.336, 12.797] | 12.135 |
| vidstg / tent_stvg | 10303 / 732 | 8.721 | 9.774 | 2.271 | 9.512 [8.928, 10.098] | 9.561 |
| vidstg / sar_stvg | 10303 / 732 | 11.129 | 12.831 | 3.300 | 12.065 [11.336, 12.797] | 12.135 |
| vidstg / dino_refine | 10303 / 732 | 13.985 | 18.024 | 6.920 | 15.372 [14.482, 16.297] | 15.473 |
| vidstg / target_trained_reference | 10303 / 732 | 19.296 | 28.137 | 13.103 | 21.449 [20.202, 22.696] | 21.518 |

Target-trained reference is supervised and not a TTA method or mathematical upper bound. TENT/SAR/EATA are STVG temporal/decoder ports. EATA HC2 supplemental uses the completed Vid-source Fisher; the unavailable EATA VidSTG direction remains user paused. Old IoU-energy Ours is excluded.

This assembler independently groups queries then parents and joins every baseline row to the new exact Frozen metric/query/order/arrival. Root still owes independent whole-table audit, all required plots/strata/negative tails, actual visual inspection, verified public publication and stage closing before later stages.
