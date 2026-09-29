# Superseded O3 KNN Critic Preference Memory

This CPU experiment finished before user steering replaced it with Conditional Online OPD. It is preserved for continuity; not rerun, promoted, or compared as a winner against the new O3 on different streams.

Original O1 heterogeneous32-source stream,8 specialist arrivals; fixed top3 oriented pair memory. Primary24 nonexpert delta tIoU: +2.0778 pp, CI [-0.4627, +5.1695]; vIoU: +0.4862 pp, CI [-1.0471, +2.7047].

23/24 choices changed; vIoU gains/losses/unchanged9/11/4,one>5pp harm. Full agreement1/24 to7/24. Both intervals cross zero. The weak positive mean does not establish reliable transfer; all cases, negative tails and references are retained.

No GPU/backbone/expert/optimizer work. Reused cached features/teacher data and O1 previously dual-checked candidate task values after online seal. Independent audit reconstructs896 pair votes and223 stored pairs, all state hashes and strictly past neighbors; maximum scalar vote error1.78e-15. Public scalar audit independently checks45 mean/interval summaries. Tests for empty memory, orientation reversal and unique top3/zero-vector behavior passed3/3 before steering.

Pair orientation is handled by storing one canonical pair and searching its nearer signed view; equivalently top absolute cosine with signed vote. No pair contributes twice. Native score only breaks win ties. Full teacher extras are consumed after online seal, task values last. No raw GT reread.

Public ROWS retains all32 selections, candidate metrics, score/win/state summaries. Full per-pair neighbor trace remains locally in ONLINE.json/ROWS.json, together with old feature caches. This differs from the current O3: different cohort/stream and mechanism; do not combine their means.
