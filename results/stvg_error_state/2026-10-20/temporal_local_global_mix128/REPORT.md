# One fixed local/global posterior combination: exposed HC128

New=.5 HiResAuxMix+.5 pure DenseLocal, same65/2080/MBR and frozen E/Live41/42/43. One combination only; no model/RGB/GPU/expert/training/student/spatial update, no OPD. All128 sealed before new GT. Repeatedly used development parents; seed average within parent then equal parents.

| Output | full vIoU percent | tIoU percent | boundary MAE frames |
|---|---:|---:|---:|
| Native | 27.928474 | 51.439514 | 77.238281 |
| HiResAuxMix | 29.534908 | 55.920522 | 70.367188 |
| DenseLocal | 28.738379 | 53.792321 | 76.757812 |
| DenseLocalMix | 28.901490 | 53.597804 | 74.714844 |
| LocalGlobalMix | 29.126845 | 54.662588 | 74.246094 |

| Paired comparison | full-v pp [CI95] | tIoU pp [CI95] | losses<-5pp |
|---|---|---|---:|
| LocalGlobalMix_minus_Native | +1.198371 [-0.22226598755204505, 2.589686548524093] | +3.223074 [0.8823942236169263, 5.469667608475808] | 17 |
| LocalGlobalMix_minus_HiResAuxMix | -0.408063 [-1.4693146372616077, 0.6182771620849865] | -1.257934 [-3.3099759267393227, 0.6988180588467621] | 19 |
| LocalGlobalMix_minus_DenseLocal | +0.388466 [-0.3612766025156, 1.07112458608405] | +0.870267 [-0.36533354795534656, 2.030487731822127] | 5 |
| LocalGlobalMix_minus_DenseLocalMix | +0.225355 [-0.3633313359470025, 0.7295091398289469] | +1.064784 [0.13756497036028584, 1.965651444119714] | 1 |

## Complementarity and severe tails

Groups use GT only AFTER sealing, never online selection. Dense-only means pure Dense improves Native but Strong does not. Signed gain and all128 contributions retained; capped-positive retention is explanatory and cannot hide negative signed effects. All group aliases and individual severe changes in COMPLEMENTARITY_PUBLIC.json.

```json
{
  "groups": {
    "both_improve": {
      "parents": 47,
      "aliases": [
        "RC006",
        "RC010",
        "RC012",
        "RC013",
        "RC015",
        "RC019",
        "RC020",
        "RC023",
        "RC024",
        "RC025",
        "RC027",
        "RC028",
        "RC029",
        "RC034",
        "RC039",
        "RC041",
        "RC043",
        "RC045",
        "RC052",
        "RC053",
        "RC059",
        "RC060",
        "RC065",
        "RC066",
        "RC072",
        "RC073",
        "RC079",
        "RC081",
        "RC084",
        "RC088",
        "RC092",
        "RC094",
        "RC097",
        "RC098",
        "RC102",
        "RC103",
        "RC105",
        "RC107",
        "RC108",
        "RC114",
        "RC115",
        "RC119",
        "RC120",
        "RC121",
        "RC122",
        "RC124",
        "RC126"
      ],
      "Dense_minus_Native": {
        "group_mean_pp": 7.541221536486598,
        "contribution_to_all128_mean_pp": 2.7690422829286727
      },
      "Strong_minus_Native": {
        "group_mean_pp": 6.509098331138077,
        "contribution_to_all128_mean_pp": 2.3900595434647625
      },
      "New_minus_Native": {
        "group_mean_pp": 7.46166361447027,
        "contribution_to_all128_mean_pp": 2.7398296084383023
      },
      "New_minus_Strong": {
        "group_mean_pp": 0.952565283332193,
        "contribution_to_all128_mean_pp": 0.34977006497353963
      },
      "Dense_positive_gain_sum_pp": 354.4374122148701,
      "capped_positive_Dense_gain_retained_fraction": 0.8743108107051244,
      "Dense_gain_sign_retained_parents": 47
    },
    "Dense_only": {
      "parents": 24,
      "aliases": [
        "RC003",
        "RC004",
        "RC018",
        "RC031",
        "RC033",
        "RC038",
        "RC046",
        "RC049",
        "RC050",
        "RC055",
        "RC056",
        "RC057",
        "RC062",
        "RC064",
        "RC071",
        "RC083",
        "RC086",
        "RC087",
        "RC089",
        "RC090",
        "RC095",
        "RC099",
        "RC101",
        "RC104"
      ],
      "Dense_minus_Native": {
        "group_mean_pp": 5.103799405843512,
        "contribution_to_all128_mean_pp": 0.9569623885956585
      },
      "Strong_minus_Native": {
        "group_mean_pp": -1.974146238471344,
        "contribution_to_all128_mean_pp": -0.370152419713377
      },
      "New_minus_Native": {
        "group_mean_pp": 2.2332561965440583,
        "contribution_to_all128_mean_pp": 0.4187355368520109
      },
      "New_minus_Strong": {
        "group_mean_pp": 4.207402435015402,
        "contribution_to_all128_mean_pp": 0.7888879565653878
      },
      "Dense_positive_gain_sum_pp": 122.49118574024429,
      "capped_positive_Dense_gain_retained_fraction": 0.4196519995175745,
      "Dense_gain_sign_retained_parents": 20
    },
    "Strong_only": {
      "parents": 25,
      "aliases": [
        "RC002",
        "RC005",
        "RC011",
        "RC016",
        "RC017",
        "RC021",
        "RC022",
        "RC030",
        "RC032",
        "RC035",
        "RC036",
        "RC037",
        "RC044",
        "RC061",
        "RC068",
        "RC069",
        "RC078",
        "RC085",
        "RC091",
        "RC096",
        "RC100",
        "RC118",
        "RC123",
        "RC125",
        "RC128"
      ],
      "Dense_minus_Native": {
        "group_mean_pp": -5.628541774753383,
        "contribution_to_all128_mean_pp": -1.09932456538152
      },
      "Strong_minus_Native": {
        "group_mean_pp": 2.0259056997151625,
        "contribution_to_all128_mean_pp": 0.3956847069756177
      },
      "New_minus_Native": {
        "group_mean_pp": -3.045683983069391,
        "contribution_to_all128_mean_pp": -0.5948601529432405
      },
      "New_minus_Strong": {
        "group_mean_pp": -5.071589682784554,
        "contribution_to_all128_mean_pp": -0.9905448599188582
      },
      "Dense_positive_gain_sum_pp": 0.0,
      "capped_positive_Dense_gain_retained_fraction": null,
      "Dense_gain_sign_retained_parents": 4
    },
    "neither": {
      "parents": 32,
      "aliases": [
        "RC001",
        "RC007",
        "RC008",
        "RC009",
        "RC014",
        "RC026",
        "RC040",
        "RC042",
        "RC047",
        "RC048",
        "RC051",
        "RC054",
        "RC058",
        "RC063",
        "RC067",
        "RC070",
        "RC074",
        "RC075",
        "RC076",
        "RC077",
        "RC080",
        "RC082",
        "RC093",
        "RC106",
        "RC109",
        "RC110",
        "RC111",
        "RC112",
        "RC113",
        "RC116",
        "RC117",
        "RC127"
      ],
      "Dense_minus_Native": {
        "group_mean_pp": -7.267100057355303,
        "contribution_to_all128_mean_pp": -1.8167750143388257
      },
      "Strong_minus_Native": {
        "group_mean_pp": -3.236630893933144,
        "contribution_to_all128_mean_pp": -0.809157723483286
      },
      "New_minus_Native": {
        "group_mean_pp": -5.461336700798706,
        "contribution_to_all128_mean_pp": -1.3653341751996766
      },
      "New_minus_Strong": {
        "group_mean_pp": -2.224705806865563,
        "contribution_to_all128_mean_pp": -0.5561764517163907
      },
      "Dense_positive_gain_sum_pp": 0.0,
      "capped_positive_Dense_gain_retained_fraction": null,
      "Dense_gain_sign_retained_parents": 3
    }
  },
  "severe": {
    "reference": "delta to Native, original DenseLocalMix severe16; no online grouping",
    "old_severe": 16,
    "any_alleviated": 4,
    "exited_severe": 3,
    "persistent_severe": 13,
    "worsened": 9,
    "unchanged": 3,
    "new_severe_outside_old16": 4,
    "new_total_severe": 17,
    "old_aliases": [
      "RC007",
      "RC014",
      "RC017",
      "RC026",
      "RC030",
      "RC036",
      "RC042",
      "RC069",
      "RC076",
      "RC080",
      "RC082",
      "RC100",
      "RC111",
      "RC118",
      "RC125",
      "RC128"
    ],
    "exited_aliases": [
      "RC017",
      "RC036",
      "RC128"
    ],
    "persistent_aliases": [
      "RC007",
      "RC014",
      "RC026",
      "RC030",
      "RC042",
      "RC069",
      "RC076",
      "RC080",
      "RC082",
      "RC100",
      "RC111",
      "RC118",
      "RC125"
    ],
    "new_aliases": [
      "RC009",
      "RC051",
      "RC077",
      "RC110"
    ],
    "per_old_severe": [
      {
        "alias": "RC007",
        "old_delta_pp": -5.158170905291848,
        "new_delta_pp": -5.158170905291848,
        "new_minus_old_pp": 0.0
      },
      {
        "alias": "RC014",
        "old_delta_pp": -14.27661553020802,
        "new_delta_pp": -16.036281537049824,
        "new_minus_old_pp": -1.7596660068418046
      },
      {
        "alias": "RC017",
        "old_delta_pp": -5.763245276773099,
        "new_delta_pp": -4.9072939548994166,
        "new_minus_old_pp": 0.8559513218736825
      },
      {
        "alias": "RC026",
        "old_delta_pp": -5.474031129869095,
        "new_delta_pp": -8.113373657196938,
        "new_minus_old_pp": -2.639342527327843
      },
      {
        "alias": "RC030",
        "old_delta_pp": -10.766417020905966,
        "new_delta_pp": -10.766417020905966,
        "new_minus_old_pp": 0.0
      },
      {
        "alias": "RC036",
        "old_delta_pp": -5.196838793431252,
        "new_delta_pp": -3.7387862318252862,
        "new_minus_old_pp": 1.4580525616059656
      },
      {
        "alias": "RC042",
        "old_delta_pp": -32.4818528887483,
        "new_delta_pp": -32.78718234894906,
        "new_minus_old_pp": -0.3053294602007597
      },
      {
        "alias": "RC069",
        "old_delta_pp": -5.124333374040635,
        "new_delta_pp": -5.124333374040635,
        "new_minus_old_pp": 0.0
      },
      {
        "alias": "RC076",
        "old_delta_pp": -5.0498349864849965,
        "new_delta_pp": -9.304221981513706,
        "new_minus_old_pp": -4.254386995028709
      },
      {
        "alias": "RC080",
        "old_delta_pp": -22.46286601594561,
        "new_delta_pp": -23.95030607760755,
        "new_minus_old_pp": -1.487440061661939
      },
      {
        "alias": "RC082",
        "old_delta_pp": -8.921954563120526,
        "new_delta_pp": -12.067478685782685,
        "new_minus_old_pp": -3.1455241226621595
      },
      {
        "alias": "RC100",
        "old_delta_pp": -8.188912640017332,
        "new_delta_pp": -7.253486849984575,
        "new_minus_old_pp": 0.9354257900327569
      },
      {
        "alias": "RC111",
        "old_delta_pp": -10.638717443606877,
        "new_delta_pp": -14.70384134292842,
        "new_minus_old_pp": -4.065123899321543
      },
      {
        "alias": "RC118",
        "old_delta_pp": -7.901062713906537,
        "new_delta_pp": -10.96082772690693,
        "new_minus_old_pp": -3.0597650130003924
      },
      {
        "alias": "RC125",
        "old_delta_pp": -12.886400829482517,
        "new_delta_pp": -13.605087402302734,
        "new_minus_old_pp": -0.7186865728202179
      },
      {
        "alias": "RC128",
        "old_delta_pp": -6.862232538893231,
        "new_delta_pp": -4.923209572509901,
        "new_minus_old_pp": 1.9390229663833303
      }
    ]
  },
  "grouping_uses_GT_only_after_global_seal": true,
  "no_GT_online_selection": true
}
```

## Decision / confirmation / cost

```json
{
  "development_mean_gain_over_Strong": false,
  "primary": {
    "mean": -0.40806329009632164,
    "CI95": [
      -1.4693146372616077,
      0.6182771620849865
    ],
    "parents": 128
  },
  "decision": "close_this_fixed_mixture_no_weight_search",
  "no_immediate2pp_gate": true,
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_GT_online_selection": true,
  "no_weight_sweep": true,
  "no_automatic_adaptive_observation": true,
  "confirmation_status": "not_run",
  "goal_achieved": false
}
{
  "validation_parents": 237,
  "development_parents": 128,
  "remaining_validation_parents": 109,
  "overlaps_previously_used_confirmation": 109,
  "meets_requested_minimum128": false,
  "not_untouched_method_test": true,
  "decision": "requires_explicit_roster_amendment_if_positive; do not silently shrink or change split"
}
{
  "CPU_prediction_seconds": 2.2704537350218743,
  "GPU_seconds": 0,
  "new_model_forwards": 0,
  "new_RGB": 0,
  "new_experts": 0,
  "network_training": 0,
  "spatial_updates": 0,
  "student_updates": 0,
  "not_deployment_latency": true,
  "CPU_audit_and_scoring_seconds": 20.715412108926103,
  "publication_archive_excluded": true,
  "source_cache_generation_cost_separate": 1531.9604845391586,
  "standalone_deployment_profile": "not_measured; cached combination is not full inference latency"
}
```

No2pp threshold was imposed on this single development complementarity check. A positive mean relativeStrong is a candidate signal, not statistical/independent proof. Do not sweep weights after failure. No automatic OPD or adaptive observation. Confirmation requires a source-disjoint prelocked panel and separate actual deployment profiling; the old1532s matrix is not deployment latency. Native/Strong/Dense/DenseMix metrics reproduced, all negative/flat parents and no-reference states retained. Expected-quality numbers are linear combinations of old offline GT diagnostic values, not fresh model feedback or a gate. No spatial/CURRENT/MAIN/PAPER change.

## Explanation field correction001 (no prediction/score change)

Canonical group labels are in COMPLEMENTARITY_CORRECTED_PUBLIC.json and EXPLANATION_CORRECTION_001_PUBLIC.json. The original Dense_gain_sign_retained_parents counted New-positive parents in all groups, including Strong-only4/neither3 where Dense was not positive. Corrected Dense-positive retained counts are both47/Dense-only20/Strong-only0/neither0; New-positive counts remain47/20/4/3. Original bytes are retained, and membership/means/contributions/capped retention fractions/main results are unchanged.

The mixture has real case-level complementarity:24 Dense-only parents, new retains positive sign in20 and capped41.9652% of their Dense gains, contributing+0.788888pp overStrong to the all128 mean. However25 Strong-only parents contribute-0.990545pp versusStrong, neither32 contribute-0.556176pp, both47 contribute+0.349770pp; sum-0.408063pp. These GT-defined groups do not furnish an online policy. Old16 Native-relative severe cases:4 alleviate (3 exit),9 worsen,3 unchanged;13 persist and4 new cases produce17 severe. Do not confuse Native-relative severe migration with severe losses versusStrong.

Under the declared no2pp-gate rule the mixture does not pass even the mean-overStrong condition; no independent confirmation or standalone deployment GPU profile is run. DenseLocal remains a candidate information source, while this one fixed mixture closes. No weight sweep, no automatic adaptive-observation/OPD; current evidence does not show when/where state-guided observation would avoid the observed harms. Cached cost2.270454s prediction+20.715412s CPU audit/scoring is not full deployment latency.

## Baseline tails with a common Native reference

| Output | full vIoU percent | tIoU percent | losses<-5pp vs Native |
|---|---:|---:|---:|
| Native | 27.928474 | 51.439514 | 0 |
| HiResAuxMix | 29.534908 | 55.920522 | 12 |
| DenseLocal | 28.738379 | 53.792321 | 23 |
| DenseLocalMix | 28.901490 | 53.597804 | 16 |
| LocalGlobalMix | 29.126845 | 54.662588 | 17 |

New also has19 losses<-5pp relativeStrong, which is a separate comparison from17 relativeNative.
