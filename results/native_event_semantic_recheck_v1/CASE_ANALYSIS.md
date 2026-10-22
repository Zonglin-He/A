# What transferred and what failed

The new learned semantic coordinates provide a stronger HC64 signal than prior token-score transformations. Primary gains over matched-histogram random coordinates and time reversal have positive paired intervals. Its two-round update exceeds the one-round arm, while using the same second-round feedback without native prior performs worse. This supports a useful interaction between the native temporal prior and learned semantic representation on this exposed panel; it does not establish transfer.

The unchanged HC32/Vid32 recheck fails. The compact native temporal-memory interface restores all64 original head input and sigmoid output arrays bitwise before the recheck. No new fusion, visual encoder, text or expert inference is used. Therefore changing the feature interface is not an explanation for that failure.

Postseal examples: HC2_RECHECK014 loses18.67tpp while expanding by2.20s; event-vs-before feedback AUC.467 versus event-vs-after.899. HC2_RECHECK011 loses9.34tpp while contracting by3.48s; before/after semantic mean separation exceeds event/outside separation. VIDSTG_RECHECK017 loses37.14tpp by shortening13.01s; VIDSTG_RECHECK025 gains51.54tpp by extending13.81s. These are offline examples, not online duration/quality gates. The same update can help or damage either boundary; no GT-selected action or state is deployed.

A prospective three-state test used the same semantic features and two rounds to distinguish before/event/after. It failed the fixed panels, particularly Vid32, where both time and complete-v paired intervals are negative. A stronger HC64 unordered-background control also fails to transfer; it is not promoted. Background mixing alone is not an established root cause. [All three-state results](../native_event_three_state_v1/ANALYSIS.md).

The new source-only correlation audit shows34/60 class coordinates varying in effectively2.1–2.8 covariance dimensions; Vid32 median mean-absolute class correlation.564. This makes the independent-class pseudo-likelihood assumption questionable, but does not prove it caused the failures. A separately locked covariance-aware likelihood can test that hypothesis; it has not been run as part of these reports. No further iterations, strength scan or control selection is justified by these results.

[Postseal outcome and feature audit](MECHANISM_AUDIT.json), [source-only correlation audit](CORRELATION_AUDIT.json), [all signed recheck outcomes](signed.tsv), [all severe declines](severe_declines.tsv). All panels are previously exposed development evidence.
