# Fixed same-input128-parent expert comparison: vidstg

Same original Source and eight indices, current ReferDINO versus official ReferFormer R50. Each expert payload is acquired once/query and shared by its simple interpolation and structured inference. Both simple rows use original native Source time; each Ours independently rebuilds that expert masks/geometry/E ROI features/prototypes/posterior/own interval/finalDP. No new Source acquisitions or full12205 completion. Same observation count is not equal compute.

|Method|tIoU %|vIoU %|sIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|---:|
|Source|32.121535|13.322903|36.052955|16.406250|3.906250|
|CurrentSimple|32.121535|13.867903|36.443529|17.968750|7.812500|
|CurrentOurs|35.202435|15.339259|38.392840|19.531250|10.156250|
|AlternativeSimple|32.121535|14.299784|38.398028|18.750000|7.031250|
|AlternativeOurs|34.603561|15.639079|38.892904|20.312500|10.156250|

Between-expert gain difference:

{
  "v": {
    "mean": -0.13206024604491368,
    "CI95": [
      -1.2558989122875994,
      0.9711856242929763
    ],
    "queries": 128,
    "parents": 128,
    "improved": 44,
    "harmed": 60,
    "unchanged": 24,
    "declined_over5pp": 11
  },
  "t": {
    "mean": -0.5988736726427317,
    "CI95": [
      -1.6325655872790064,
      0.29247820631324334
    ],
    "queries": 128,
    "parents": 128,
    "improved": 23,
    "harmed": 26,
    "unchanged": 79,
    "declined_over5pp": 12
  },
  "s": {
    "mean": -1.4544352185105507,
    "CI95": [
      -3.9915672935214515,
      1.0029688958913001
    ],
    "queries": 128,
    "parents": 128,
    "improved": 54,
    "harmed": 71,
    "unchanged": 3,
    "declined_over5pp": 35
  },
  "vIoU_ge_03": {
    "mean": 0.0,
    "CI95": [
      -4.6875,
      4.6875
    ],
    "queries": 128,
    "parents": 128,
    "improved": 4,
    "harmed": 4,
    "unchanged": 120,
    "declined_over5pp": 4
  },
  "vIoU_ge_05": {
    "mean": 0.78125,
    "CI95": [
      -2.34375,
      3.90625
    ],
    "queries": 128,
    "parents": 128,
    "improved": 3,
    "harmed": 2,
    "unchanged": 123,
    "declined_over5pp": 2
  },
  "observed_actual_sIoU": {
    "mean": 1.8456925812776765,
    "CI95": [
      -0.8243779514777692,
      4.793070574869265
    ],
    "queries": 103,
    "parents": 103,
    "improved": 45,
    "harmed": 45,
    "unchanged": 13,
    "declined_over5pp": 16
  },
  "unobserved_actual_sIoU": {
    "mean": -1.7712792676348275,
    "CI95": [
      -4.606123542457368,
      0.8935874356021615
    ],
    "queries": 128,
    "parents": 128,
    "improved": 54,
    "harmed": 70,
    "unchanged": 4,
    "declined_over5pp": 39
  }
}

All within-expert/between-expert paired CIs, signed positives/negatives, severe union/rescues and short/long groups accompany the report. No target-score config selection or unsupported expert-independence claim. Actual saved expert/conditional numeric costs are separate from inherited Source cost, not summed into uncached deployment.
