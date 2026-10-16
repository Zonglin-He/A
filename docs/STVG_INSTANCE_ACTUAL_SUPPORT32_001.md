# Actual gradient support versus fixed observation support

The original five-arm scoring uses the same four pre-fixed observation positions for every arm. Its inherited column names supervised/unsupervised therefore mean common observation positions versus their known-GT complement. Some positions select Source, are rejected or have no usable anchor. They should not be described as four actual gradient-supervised frames. The complement includes off-Source-grid known GT, as defined by the original official/independent scorer.

This additional postseal diagnostic instead uses each updater's actual positive-weight gradient-anchor positions. It compares Student and corrected Temporal on exactly that same subset and its complete known-GT complement. All vIoU, tIoU and dense sIoU are asserted exactly unchanged. No new Teacher/Student fit, protocol selection or Bootstrap occurs; all nulls and all96 source/updater rows remain.

For E1, HC actual-anchor sIoU changes+1.268pp on11 known sources(5 unknown), complement+0.055pp on all16. Vid actual-anchor changes+3.528pp on6 known sources(10 unknown), complement+2.516pp on all16. These subset means are not directly interchangeable or a guarantee of propagation safety. Negative and zero changes are retained.

Development gain concentration also limits interpretation. HC E1-E0 becomes-.088pp if its largest positive source is omitted. Vid E1-E2 becomes+.000516pp without its largest positive source. These are descriptive arithmetic, not a new statistical test or changed advancement rule. The original mean-based finite development rule remains locked; the fixed64 confirmation continues unchanged. A small positive development mean does not establish stable updater benefit.

Main32 results and all old predictions stay immutable. This supplement corrects metric interpretation and adds actual-support measurements, not a replacement method score. Source media, GT coordinates, tensors, weights, Query-caption and raw Teacher responses remain private.
