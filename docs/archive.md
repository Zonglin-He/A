# Research archive and runtime separation

The complete pre-cleanup public tree is preserved at commit
`c310af5888dca78d6ea7b0a280cb67d209a90337` on
[archive/research-pre-mainline-20261022](https://github.com/Zonglin-He/A/tree/archive/research-pre-mainline-20261022).
It contains all historical source, protocols, development results, signed negative
findings and cost studies. The clean main and research branches retain that commit
as an ancestor. No history rewrite or deletion of private evidence is part of cleanup.

Current main contains one maintained inference method and its required dependencies.
Research scripts/configurations, old OPD/Direct policies, conditional replay, 320/448
trial launchers and historical result trees are not installed or reachable here.
The frozen upstream TA-STVG and ReferDINO code is preserved intact because the model
loaders require it; upstream training utilities do not register a new project method.

Locally, stvg-mainline is the maintained clean tree. The completed stvg-opd
checkout and its linked private assets stay in place as the frozen full-run archive.
Its old code fingerprint must remain unchanged. Do not run cleanup inference concurrently
or use this refactor to resume its old receipts. Import the final public reports into
this clean tree after both domains seal/score, then publish this tree rather than
re-exporting the archived research checkout.

To reproduce the cached cleanup audit against an archived runtime with its private
saved captures, run `python tools/verify_saved.py --reference-root /path/to/archived-runtime`.
This does not load a model or labels. It is optional for users without those private captures.

Source/evidence maintenance: `python tools/archive.py check`, then `snapshot`, then
`check`. The source inventory excludes itself to avoid a recursive digest and excludes
private assets. Publication receipts separately identify verified commit/tree/bytes.
