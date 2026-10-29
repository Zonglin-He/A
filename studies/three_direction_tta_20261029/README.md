# Three-direction TTA investigation, 2026-10-29–30

Three real pretrained baselines were restored on RTX 5090, modified, and evaluated with sealed predictions and independent CPU audits. Read the [consolidated report](research/REPORT.md) for numerical outcomes, failure analysis and research decisions.

- [AVIS](avis/README.md): 8 convenience videos; 128-frame short panel and 495-frame full-context panel, 14 total arms. Current late-fusion LN adaptation is not adopted.
- [GREAT](affordance/README.md): 12 fixed point-cloud parents, original and canonical-view LN controls. Parameter adaptation is not adopted; permutation ensemble has mixed signed outcomes.
- [3EED](crossplatform/README.md): 9-example LN pilot plus two distinct 36-frame geometry validations. Geometric orientation refinement improves the mean but has severe negative tails and is not network TTA.

`research/STATE.json` separates completed experiments from ongoing bulk downloads. Official 3EED data are fully hash/CRC verified; AVIS complete public test split is verified, while its full archive CRC remains pending; GREAT uses verified model tensors and a fixed evaluation subset while its bulk Unseen-aff archive continues downloading. This is a dated snapshot, not a claim of full benchmark reproduction.

Code, compatibility patches, locked protocols, scalar paired outcomes, independent auditors, failures and measurement corrections are included. Media, annotations, captions, point coordinates, masks, predictions, weights, features and credentials are excluded. Some exact execution scripts require locally acquired official data and paths; independent auditor copies have documented root-path portability edits. Leaf manifests retain original staging statuses as historical evidence.

Upstream credit: [AVIS](https://github.com/ruohaoguo/avis), [GREAT](https://github.com/yawen-shao/GREAT_code), [3EED](https://github.com/worldbench/3EED). Upstream licensing remains applicable; vendor trees are not redistributed. No claim that these generic update ingredients alone are novel.
