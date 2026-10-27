# Actual mechanism figures

The user explicitly authorized publishing these three existing figures on 2026-10-26, including the visible RGB, query captions and annotated GT overlays. This is a narrow exception to the default private-media policy; raw videos, annotations, masks, features and weights are not included. The original inspected PNG/PDF bytes are unchanged.

These are post-hoc positive-extreme illustrations, not representative samples. Green boxes are actual annotated GT only where supported. Missing annotations are marked, and Source/current/GT temporal intervals and all eight expert observations are shown. SAM2 is not included because its propagation experiment has not run.

| Case | Capability | Files |
|---|---|---|
| 1 | Between-observation recovery | [PNG](case_1.png) · [PDF](case_1.pdf) |
| 2 | Temporal-boundary recovery, with remaining spatial error retained | [PNG](case_2.png) · [PDF](case_2.pdf) |
| 3 | Same-candidate independent selection versus final trajectory DP | [PNG](case_3.png) · [PDF](case_3.pdf) |

Case 2 is a temporal illustration: its spatial trajectory remains imperfect and its unobserved fixed-support sIoU change is negative. Do not treat it as proof of correct spatial localization. The scalar/hash/selection and root visual-inspection receipts are in the parent folder.


2026-10-27 display revision: the figure header now contains only Query. Case/dataset titles, header deltas and explanatory notices were removed at the user's request. All below-header image pixels and all original image/geometry operators are exactly preserved. See results/figure_query_only_v1 for the independent edit and inspection receipts; earlier publication_authorization.json documents historical original figure bytes.
