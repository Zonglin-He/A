# Fixed same-input128-parent expert comparison: hc2

Same original Source and eight indices, current ReferDINO versus official ReferFormer R50. Each expert payload is acquired once/query and shared by its simple interpolation and structured inference. Both simple rows use original native Source time; each Ours independently rebuilds that expert masks/geometry/E ROI features/prototypes/posterior/own interval/finalDP. No new Source acquisitions or full12205 completion. Same observation count is not equal compute.

|Method|tIoU %|vIoU %|sIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|---:|
|Source|46.509478|25.124253|52.562045|42.968750|14.062500|
|CurrentSimple|46.509478|26.879620|49.660563|45.312500|15.625000|
|CurrentOurs|54.968548|31.791298|54.966026|50.781250|22.656250|
|AlternativeSimple|46.509478|23.073061|45.307994|35.937500|13.281250|
|AlternativeOurs|54.769180|27.552349|48.911149|43.750000|17.187500|

Between-expert gain difference:

{
  "v": {
    "mean": -0.4323904035632746,
    "CI95": [
      -1.65787845432942,
      0.8380892004171919
    ],
    "queries": 128,
    "parents": 128,
    "improved": 55,
    "harmed": 60,
    "unchanged": 13,
    "declined_over5pp": 24
  },
  "t": {
    "mean": -0.19936787672136774,
    "CI95": [
      -0.9209644041929725,
      0.5363202650027828
    ],
    "queries": 128,
    "parents": 128,
    "improved": 15,
    "harmed": 19,
    "unchanged": 94,
    "declined_over5pp": 13
  },
  "s": {
    "mean": -1.7023069297250566,
    "CI95": [
      -4.085042095287849,
      0.6426669467512492
    ],
    "queries": 128,
    "parents": 128,
    "improved": 63,
    "harmed": 63,
    "unchanged": 2,
    "declined_over5pp": 39
  },
  "vIoU_ge_03": {
    "mean": 2.34375,
    "CI95": [
      -3.90625,
      8.59375
    ],
    "queries": 128,
    "parents": 128,
    "improved": 9,
    "harmed": 6,
    "unchanged": 113,
    "declined_over5pp": 6
  },
  "vIoU_ge_05": {
    "mean": -3.125,
    "CI95": [
      -9.375,
      3.125
    ],
    "queries": 128,
    "parents": 128,
    "improved": 6,
    "harmed": 10,
    "unchanged": 112,
    "declined_over5pp": 10
  },
  "observed_actual_sIoU": {
    "mean": -0.2829051854856364,
    "CI95": [
      -2.7451167930199034,
      1.979487870937544
    ],
    "queries": 126,
    "parents": 126,
    "improved": 65,
    "harmed": 44,
    "unchanged": 17,
    "declined_over5pp": 19
  },
  "unobserved_actual_sIoU": {
    "mean": -2.144289331953458,
    "CI95": [
      -4.782527402668859,
      0.4492288286431176
    ],
    "queries": 128,
    "parents": 128,
    "improved": 57,
    "harmed": 69,
    "unchanged": 2,
    "declined_over5pp": 43
  }
}

All within-expert/between-expert paired CIs, signed positives/negatives, severe union/rescues and short/long groups accompany the report. No target-score config selection or unsupported expert-independence claim. Actual saved expert/conditional numeric costs are separate from inherited Source cost, not summed into uncached deployment.
