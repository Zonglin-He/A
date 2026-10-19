# Frozen policy confirmation on reserved HC32

Selected on exposed HC128 before confirmation: **N005F025**, scales[0.05, 0.25], seeds41/42/43. No configuration switching or fitting to confirmation scores. Parent roster was reserved before HC128 and not used in these rounds; historical project Source/old-method evaluation exposure remains. This is confirmation for these policy/interface selections, not globally never-seen data.

Same original8 physical positions, native ReferDINO selected trajectory, E+finalMBR, eight independent content states, frozen TA-STVG,40Adamsteps/.01/tau.05/32candidates. Source and teacher acquired once per held parent, shared across controls. All32 outputs sealed before confirmationGT. Metrics average seeds within parent, then bootstrap32 parents, not96 independent videos.

| Comparison | Full vIoU pp [95%CI] | Outside8 actual sIoU pp [95%CI] |
|---|---:|---:|
| Direct_minus_E_MBR | -0.1314 [-0.6723,+0.5011] | -0.5022 [-2.6146,+1.3129] |
| Selected_Live_seedmean_minus_E_MBR | +0.0096 [-0.4110,+0.5691] | +0.2619 [-0.6835,+1.4815] |
| Selected_Fixed_seedmean_minus_E_MBR | -0.0459 [-0.1950,+0.0969] | +0.0380 [-0.2674,+0.4102] |
| Selected_Live_seedmean_minus_Direct | +0.1410 [-0.0835,+0.4831] | +0.7641 [-0.1860,+2.4359] |
| Selected_Live_seedmean_minus_Selected_Fixed_seedmean | +0.0555 [-0.2897,+0.5260] | +0.2238 [-0.5302,+1.1559] |
| Selected_Live_seedmean_minus_Old_Live_seedmean | -0.0724 [-0.1841,+0.0227] | +0.3297 [-0.1556,+0.9935] |

## Seed 41

| Comparison | Full vIoU pp [95%CI] | Outside8 actual sIoU pp [95%CI] |
|---|---:|---:|
| Direct_minus_E_MBR | -0.1314 [-0.6723,+0.5011] | -0.5022 [-2.6146,+1.3129] |
| Selected_Live_41_minus_E_MBR | -0.0302 [-0.4786,+0.5462] | +0.3755 [-0.7784,+2.0027] |
| Selected_Fixed_41_minus_E_MBR | -0.0781 [-0.2577,+0.0925] | -0.0139 [-0.3414,+0.3731] |
| Selected_Live_41_minus_Direct | +0.1011 [-0.1009,+0.4286] | +0.8777 [-0.0899,+2.5738] |
| Selected_Live_41_minus_Selected_Fixed_41 | +0.0479 [-0.2935,+0.4942] | +0.3894 [-0.5469,+1.6784] |
| Selected_Live_41_minus_Old_Live_41 | -0.0243 [-0.1542,+0.1266] | +0.6930 [-0.1403,+1.9023] |

## Seed 42

| Comparison | Full vIoU pp [95%CI] | Outside8 actual sIoU pp [95%CI] |
|---|---:|---:|
| Direct_minus_E_MBR | -0.1314 [-0.6723,+0.5011] | -0.5022 [-2.6146,+1.3129] |
| Selected_Live_42_minus_E_MBR | +0.0708 [-0.3036,+0.5922] | +0.2710 [-0.5503,+1.2992] |
| Selected_Fixed_42_minus_E_MBR | -0.0402 [-0.1816,+0.0987] | +0.0618 [-0.2450,+0.4388] |
| Selected_Live_42_minus_Direct | +0.2022 [-0.0754,+0.5725] | +0.7733 [-0.3788,+2.6020] |
| Selected_Live_42_minus_Selected_Fixed_42 | +0.1110 [-0.1847,+0.5757] | +0.2092 [-0.4436,+0.9837] |
| Selected_Live_42_minus_Old_Live_42 | -0.0609 [-0.2212,+0.0790] | +0.2334 [-0.2020,+0.7652] |

## Seed 43

| Comparison | Full vIoU pp [95%CI] | Outside8 actual sIoU pp [95%CI] |
|---|---:|---:|
| Direct_minus_E_MBR | -0.1314 [-0.6723,+0.5011] | -0.5022 [-2.6146,+1.3129] |
| Selected_Live_43_minus_E_MBR | -0.0117 [-0.4792,+0.5815] | +0.1391 [-0.7859,+1.2597] |
| Selected_Fixed_43_minus_E_MBR | -0.0193 [-0.1595,+0.1238] | +0.0662 [-0.2325,+0.4365] |
| Selected_Live_43_minus_Direct | +0.1196 [-0.1029,+0.4626] | +0.6413 [-0.4172,+2.3755] |
| Selected_Live_43_minus_Selected_Fixed_43 | +0.0076 [-0.4065,+0.5150] | +0.0729 [-0.6941,+0.9312] |
| Selected_Live_43_minus_Old_Live_43 | -0.1319 [-0.3204,-0.0104] | +0.0626 [-0.3599,+0.5956] |

## Tails, cost and bounded conclusion

{
  "Direct": {
    "positive": 10,
    "negative": 19,
    "unchanged": 3,
    "below_minus3pp": 2,
    "below_minus5pp": 0,
    "worst5": [
      [
        -4.621171837814927,
        "RH018"
      ],
      [
        -3.8422096936633814,
        "RH013"
      ],
      [
        -0.6989957090117321,
        "RH030"
      ],
      [
        -0.5914289574647791,
        "RH007"
      ],
      [
        -0.5649500345552699,
        "RH032"
      ]
    ],
    "best5": [
      [
        0.07933067105955205,
        "RH017"
      ],
      [
        0.18769408512970304,
        "RH023"
      ],
      [
        0.2454427011448046,
        "RH012"
      ],
      [
        0.5380450322214472,
        "RH029"
      ],
      [
        7.4636014966291935,
        "RH004"
      ]
    ]
  },
  "Selected_Fixed_seedmean": {
    "positive": 13,
    "negative": 16,
    "unchanged": 3,
    "below_minus3pp": 0,
    "below_minus5pp": 0,
    "worst5": [
      [
        -1.5510021571339128,
        "RH007"
      ],
      [
        -0.5658043855462508,
        "RH032"
      ],
      [
        -0.4849012461635799,
        "RH022"
      ],
      [
        -0.3647152011397675,
        "RH013"
      ],
      [
        -0.36341178220148374,
        "RH003"
      ]
    ],
    "best5": [
      [
        0.07075937295278295,
        "RH023"
      ],
      [
        0.10502965528912112,
        "RH030"
      ],
      [
        0.15443159657825278,
        "RH012"
      ],
      [
        1.0354347958937915,
        "RH004"
      ],
      [
        1.0828890300986993,
        "RH029"
      ]
    ]
  },
  "Selected_Live_seedmean": {
    "positive": 11,
    "negative": 18,
    "unchanged": 3,
    "below_minus3pp": 1,
    "below_minus5pp": 0,
    "worst5": [
      [
        -3.1178790007762665,
        "RH013"
      ],
      [
        -1.5058258337769581,
        "RH007"
      ],
      [
        -0.9580312783448175,
        "RH019"
      ],
      [
        -0.7888376035454225,
        "RH032"
      ],
      [
        -0.6987411794113357,
        "RH030"
      ]
    ],
    "best5": [
      [
        0.07546982997655993,
        "RH028"
      ],
      [
        0.1658378765320878,
        "RH012"
      ],
      [
        0.2204778743334817,
        "RH023"
      ],
      [
        1.5983378440171927,
        "RH029"
      ],
      [
        7.1138208928775954,
        "RH004"
      ]
    ]
  },
  "Old_Live_seedmean": {
    "positive": 11,
    "negative": 18,
    "unchanged": 3,
    "below_minus3pp": 0,
    "below_minus5pp": 0,
    "worst5": [
      [
        -2.3059751186925466,
        "RH013"
      ],
      [
        -0.7585616221955943,
        "RH032"
      ],
      [
        -0.7116110112481422,
        "RH018"
      ],
      [
        -0.6819190670512815,
        "RH030"
      ],
      [
        -0.4660490520241506,
        "RH022"
      ]
    ],
    "best5": [
      [
        0.06426865026137318,
        "RH028"
      ],
      [
        0.19784171777504708,
        "RH023"
      ],
      [
        0.2109481380395406,
        "RH012"
      ],
      [
        1.4665684709087203,
        "RH029"
      ],
      [
        7.507423037756519,
        "RH004"
      ]
    ]
  }
}

GPUallocation655.596s; new Sourcequeries32/offsetforwards64; native ReferDINOqueries32/imageframes256. No extra observations per video, expert models, learned heads or pretrained-weight training. Actual state updates12800. Prior HC128 cache costs excluded from new development but not free.

Direction and uncertainty are reported for selected Live−E,−Direct,−freshFixed,−oldLive. A more negative or uncertain confirmation is retained, not used to switch to another scale pair. All positive/negative/unknown parents and seeds remain. End structure iteration with the prelocked selection; CURRENT/MAIN/PAPER not promoted.

## Final measured judgement

The predeclared development ranking selected **near0.05/far0.25**. This is kept frozen; confirmation was not used to switch to another scale or the old policy. The research structure selection is finished, without production promotion.

HC128 development: selectedLive−E full vIoU +0.678401pp [0.162201,1.300158], outside8 actual sIoU +1.574997pp [0.407723,2.954757]. Relative to oldLive: +0.160732pp [-0.027718,0.472484] / +0.436425pp [-0.016150,1.105983]. Relative to Direct: +0.215706pp [-0.016843,0.550397] / +0.525699pp [0.071236,1.172937]. Relative to new freshFixed: +0.459134pp [0.061434,0.974800] / +1.048222pp [0.132353,2.130550]. Development averages improve, but full-vIoU superiority over oldLive/Direct remains uncertain. RC074 contributes +16.833099pp of new−old; excluding it ONLY as a sensitivity diagnostic leaves the other127 mean +0.029454pp. No exclusion changes the reported128-parent result.

Frozen HC32 confirmation: selectedLive−E full vIoU +0.009622pp [-0.410969,0.569150], outside8 actual sIoU +0.261879pp [-0.683549,1.481530]. SelectedLive−oldLive: −0.072352pp [-0.184114,0.022729] / +0.329681pp [-0.155555,0.993483]. SelectedLive−Direct: +0.140981pp [-0.083535,0.483091] / +0.764104pp [-0.186017,2.435943]. SelectedLive−new freshFixed: +0.055512pp [-0.289710,0.525974] / +0.223830pp [-0.530150,1.155858]. ALL these confirmation intervals cross zero. New−old full-vIoU means are negative in all three confirmation seeds41/42/43 (−0.024254/−0.060890/−0.131913pp). The development policy increment was not confirmed; these results do not establish stable independent OPD benefit over Direct/freshFixed/oldLive.

Complete spatial contribution is measured separately at the SAME frozenMBR time. Development selectedLive−Source +2.425162pp [0.530687,4.328409] full vIoU includes E−Source +1.746761pp [-0.083132,3.585545]. Confirmation selectedLive−Source +3.379444pp [-2.930257,9.051985] includes E−Source +3.369822pp [-3.044336,9.056969]. Thus total-spatial point gains of2–3pp must not be attributed to the new OPD; the confirmation total itself is uncertain. Source_MBR is no spatial adaptation, not the original native-time Source.

All seeds/parents/tails remain. Development selectedLive−E improves80, degrades40, leaves8 unchanged;4 losses exceed3pp and2 exceed5pp (RC067−6.094566, RC036−5.084032). Confirmation improves11, degrades18, leaves3 unchanged; worst RH013−3.117879pp, best RH004+7.113821pp. No output/parent/seed/best-step gates, rescues or new parameter search. Three-seed means are within-parent METRICS, not box ensembles or384/96 independent parents. Confirmation was reserved from these rounds but has earlier project Source/old-method exposure.

The policy change is real: true normalized4D mixture log-density, detached per-round size covariance/actions/targets, geometry-constructed weights, current student means and new candidate batches; no old single-Gaussian analytic KL/control-variate reused. It changes coordinates/exploration and their consistent fitting objective together, so gains cannot be assigned to coverage alone. Fixed retains initial mean/D0 in both proposal and density; Live refreshes mean and detached size context. Their contrast does not separately isolate mean refresh versus covariance-context refresh, and no importance-corrected off-policy claim is made. This is an output-box policy with actual full decoder reruns, not independently decoded candidate trajectories.

New development GPU allocation3223.816253s; frozen confirmation655.595783s; total3879.412036s (64.656867min). New development Source/ReferDINO/DINO calls0; confirmation32 Sourcequeries/64offsetforwards and32 nativeReferDINOqueries/256 imageframes, shared by controls. Actual temporary state updates92160+12800=104960; no pretrained-weight/new-head training. Confirmation mean fit times Direct1.221692s, oldLive1.262329s, selectedFixed1.282313s, selectedLive1.287289s; these are fit timings with existing cached features, not full from-scratch deployment latency. E's prior DINO evidence and prior feature/teacher acquisition are not free.

Both independent saved-artifact audits passed sampling/density/gradient/optimizer/readout/evaluator/bootstrap checks; no claim of a second complete GPU replication. No main/confirmation model run failed or was repeated. The legacy Source-cache preflight used the wrong adjacent-receipt adapter once; it was corrected to verify the original frozen PLAN/global SHA before loading, without model/GT actions or output changes. The main audit's legacy native_chart counter is unused; actual corrected-chart comparison is asserted1e−12 and independently summarized at <=4.44e−16.

Decision: archive the selected(.05,.25) version and all controls as this finite structural comparison. It has NOT established the requested stable large OPD contribution and does not justify replacing the old research baseline or deploying a new default. Do not retune on HC32, switch configurations after its outcome, or automatically start another interface/policy experiment. Original CURRENT/MAIN/PAPER registries remain unchanged.


[Source spatial comparison at fixedMBR](SOURCE_BASELINE_REPORT.md)
