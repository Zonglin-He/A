# All-source P0 failure attribution

These are postseal descriptive associations, not interventions or new parameter selections. Every positive, negative and no-update row is retained. Quality cutpoints are .3/.5 IoU; duration/motion strata use outcome-independent target-feature quartiles. Figures/cases do not replace the full matched efficacy estimates.

## hc2

| Expert quality | Parents | Total ΔvIoU pp | Current pp | Inherited pp | Teacher vs uniform GT IoU pp |
|---|---:|---:|---:|---:|---:|
| missing_GT_evaluable_expert | 25 | +0.171 | -0.051 | +0.222 | n/a |
| expert_IoU_below_0.3 | 24 | -6.574 | -8.367 | +1.793 | -0.134 |
| expert_IoU_0.3_to_0.5 | 15 | +0.386 | -1.627 | +2.013 | +0.525 |
| expert_IoU_at_least_0.5 | 64 | +2.934 | +0.777 | +2.157 | +0.908 |

Empty support: 6/256 arrivals; zero-backward arrivals: 14/256. Expert-reward gain with negative current vIoU: 94/256. Neither time readout nor optimizer configuration changed.

## vidstg

| Expert quality | Parents | Total ΔvIoU pp | Current pp | Inherited pp | Teacher vs uniform GT IoU pp |
|---|---:|---:|---:|---:|---:|
| missing_GT_evaluable_expert | 48 | +0.662 | +0.200 | +0.462 | n/a |
| expert_IoU_below_0.3 | 17 | -2.717 | -4.156 | +1.439 | -0.049 |
| expert_IoU_0.3_to_0.5 | 3 | +3.409 | +3.143 | +0.267 | -0.243 |
| expert_IoU_at_least_0.5 | 60 | +8.327 | +5.206 | +3.121 | +1.346 |

Empty support: 48/256 arrivals; zero-backward arrivals: 52/256. Expert-reward gain with negative current vIoU: 47/256. Neither time readout nor optimizer configuration changed.

