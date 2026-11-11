# R72 bounded iteration record

Claim under test: calibrated RGB/LiDAR surface support plus literal target/anchor requirements can produce candidate negatives that improve frozen-box grounding; native matching updates should be compared with identical-evidence direct reading.

Boundary: current Waymo-only model, original A256, exposed R69 VAL1280/TRAIN1024 and DEV336, no source training, no expert feedback, no FINAL access, no search across thresholds or update locations. CLIP is a cached external system reference only.

Observed predecessor: R55 rule conflicts and R51–R53 box-based RGB measurements lacked robust identity evidence. Distinct intervention: build global local-normal/depth/RGB-boundary surface components before associating them with predicted candidates; do not infer complete object geometry from visible surfaces. Surface purity and semantic identity remain unproven.

Execution: original design and raw evidence sealed; primary-paper annotation convention checked; separately lock camera-frame repair and conservative chained-role UNKNOWN rule before any task lookup; run actual per-input native matching updates for accepted negatives; seal every probability/box selection; independent numerical/raw-subset audit; join saved candidate IoUs; independent scalar and descriptive audit. All inactive/UNKNOWN/negative observations remain in denominators.

Finding: five accepted negative-bearing inputs, zero rejected source winners, no relational exclusions. Hard removal cannot replace source winners. Actual optimizer changes weights and probabilities; fixed Adam also overshoots on some steps. Exact same-loss distribution optimization and hard removal produce no task gain, so optimizer completion alone cannot provide decision-relevant evidence missing from accepted sets.

Decision: completed, valid within the declared fixed-candidate/cached-prefix protocol; claim of useful overall expert-free negative supervision weakened in this implementation. Do not adopt or sweep this recipe. This is not a disproof of all expert-free TTA, not a reproduction of GSC/DARTH/ORD, and not independent blind testing. No follow-on experiments are queued.

Preserved repairs: overly literal camera-frame gate (pre-GT), prediction validator treating raw entity scores as normalized candidates (pre-GT), descriptive NumPy boolean Counter saturation (postscore; no prediction changes). Initial evidence, failed attempts and numerical tails remain archived privately; public export removes raw identities and data.
