# Current paper source and run entry

The paper version selected on2026-10-17 is `stvg_t2_b_episodic_20261017.tex`, bound to `methods/paper/STVG_T2_B_EPISODIC_V1.json` and `scripts/run_stvg_paper_t2_b_v1.py`.

`methods/CURRENT_METHOD.json` remains the historical deployed DINO/Native-WHEN/online-LN method. It is not the paper entry. Older `.tex` files are historical drafts and preserve their original experiment context; their results and claims must not be silently relabeled as this version.

The new method is T2 Caption–Query Temporal-Lite, fixed B evidence, independent spatial Caption acquisition, actual sparse R1 references, existing fixed-sigma Gaussian OPD and original response filter, full decoded reference acceptance/rollback, and episodic Source reset. B's original internal M2 label is the existing response-filter implementation; it is not the separately proposed Decoder-Aligned constrained target solver. T2 is a human simplicity choice, not an established noninferiority result. Main integrated T2+B results remain pending until actual complete predictions, root audit, postseal scoring and public closure.
