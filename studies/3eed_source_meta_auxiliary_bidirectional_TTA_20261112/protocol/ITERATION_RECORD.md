# R78 source meta-auxiliary controlled prototype

Claim: source-trained geometry-local bidirectional modules can learn a useful unlabeled adaptation direction when the outer source objective evaluates post-update grounding. Competing explanation: ordinary extra source training/static improvement or a weaker initialization could explain apparent gains.

Fixed: R76 bidirectional epoch6; all original3EED/A256/anchors/decoder geometry frozen;338343 source-updated new-module parameters;102498 test-updated parameters;512 source TRAIN/128 exposed VAL;seed0;4 epochs per branch;identical sample/mask order;AdamW1e-4 outer learning rate. Meta outer objective0.5ground0+0.5ground1+aux versus joint ground0+aux; grounding total coefficient1 in both. Auxiliary definitions and physical support unchanged.

Inner update: exactly one functional SGD step, alpha0.5, differentiable global clip1. Alpha fixed before training from prior source gradient/step scale, not from target labels; no search. The same operator is used in source training, source validation and target inference. No detached inner gradient in meta training. Two source TRAIN inputs test nonzero post-grounding gradient to auxiliary-only heads, zero under detached-gradient control, and functional-versus-real SGD output identity. Frozen original suffix remains differentiable with respect to new input features.

Budget is matched source exposure and outer updates, not equal compute. Meta has extra suffix forward/backward and higher-order auxiliary backward; report counts and elapsed costs. Source-only shared epoch is chosen by equal-branch mean post-update VALmIoU, earliest tie; each branch tested both zero and one step. No target method selection.

Target fixed twelve-arm DEV336 comparison includes J0/J1/M0/M1, each same-information three-observation direct mean, each wrong auxiliary correspondence, originalsource/ABC/R59 and R76 frozen bidirectional. Predictions are sealed and native saved-state audited before old candidate-IoU label join. All positives, negatives, unknowns and scene results retained. Historical exposure disclosed. FINAL excluded.

Expansion only after genuine target increment beyond own frozen/direct-use and fair joint controls, with a separate locked configuration. No new loss/threshold/mask/parameter-scope grid or cherry-picking. If the tested recipe fails, stop it and distinguish source training/auxiliary alignment from the upper-level TTA research route.
