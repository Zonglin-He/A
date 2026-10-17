# Storage retirement, 2026-10-17

Explicit human authorization to delete unused artifacts/downloaded datasets/model weights, then specifically old tuning and DECoTA material. No new inference, model update, GT evaluation, promotion or restored historical queue. This receipt describes storage availability, not scientific performance.

| Batch | Original containers | Logical GiB retired | Actual available-space increase GiB |
|---|---:|---:|---:|
|Unused downloads, inactive weights, old Teacher cache|2105|95.997|95.992|
|Legacy DECoTA optimizer states and tuning feature views|94238|32.134|18.947|

The first batch removes unused MeViS/SportsGrounding/VidOR-training/GroundingYouTube archive packages, inactive VideoChat-R1.5 weight shards, abandoned STCAT fragments and closed Native/paper Teacher native.npz encodings. Original source/official receipts and native-cache retirement markers remain. Current R1/DINO/TA-STVG/TubeDETR weights, original HC/Vid videos, current Source prefix, context images, Teacher responses, predictions and research reports remain. Deleted model/dataset packages require re-download; retired Teacher features require reconstruction.

The second batch retires93156 old DECoTA baseline NPZ containers after copying every Before.npy, After.npy and metadata.npy byte-for-byte into private prediction capsules; only state_before.npy/state_after.npy are removed. It also retires1082 completed tuning H_cache containers after retaining prediction and all other fields except feature views, with tensor dtype/shape/value and nested field equality checks. All retained capsule hashes and archive members are independently read back before deleting any original container. The private archives and complete manifests stay local, excluded from public export. Original old parameter-update replays now require reconstruction; final predictions, scalar reports, negative findings and configs remain.

Reported actual free-space change accounts for archive cost, hard links and concurrent file writes; logical bytes alone are not claimed as physically freed. Two open-FD/mmap checks, per-file identity/SHA checks, CURRENT/paper/active-scope hashes and archive ledger validation accompany the retirement. Old binary paths are unavailable and mapped to retained archive members in the private retirement manifest. No claim is made that original optimizer-state replay remains immediately available.
