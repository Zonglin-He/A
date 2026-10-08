# All-source P0 failure attribution

These are postseal descriptive associations, not interventions or new parameter selections. Every positive, negative and no-update row is retained. Quality cutpoints are .3/.5 IoU; duration/motion strata use outcome-independent target-feature quartiles. Figures/cases do not replace the full matched efficacy estimates.

## hc2

| Expert quality | Parents | Total ΔvIoU pp | Current pp | Inherited pp | Teacher vs uniform GT IoU pp |
|---|---:|---:|---:|---:|---:|
| missing_GT_evaluable_expert | 25 | +0.233 | -0.142 | +0.376 | n/a |
| expert_IoU_below_0.3 | 24 | -4.309 | -5.893 | +1.584 | -0.073 |
| expert_IoU_0.3_to_0.5 | 15 | +1.475 | -0.613 | +2.088 | +0.100 |
| expert_IoU_at_least_0.5 | 64 | +3.871 | +1.549 | +2.322 | +0.181 |

Empty support: 6/256 arrivals; zero-backward arrivals: 15/256. Expert-reward gain with negative current vIoU: 90/256. Neither time readout nor optimizer configuration changed.

## vidstg

| Expert quality | Parents | Total ΔvIoU pp | Current pp | Inherited pp | Teacher vs uniform GT IoU pp |
|---|---:|---:|---:|---:|---:|
| missing_GT_evaluable_expert | 48 | +0.662 | +0.200 | +0.462 | n/a |
| expert_IoU_below_0.3 | 17 | -2.717 | -4.156 | +1.439 | -0.049 |
| expert_IoU_0.3_to_0.5 | 3 | +3.409 | +3.143 | +0.267 | -0.243 |
| expert_IoU_at_least_0.5 | 60 | +8.327 | +5.206 | +3.121 | +1.346 |

Empty support: 48/256 arrivals; zero-backward arrivals: 52/256. Expert-reward gain with negative current vIoU: 47/256. Neither time readout nor optimizer configuration changed.

