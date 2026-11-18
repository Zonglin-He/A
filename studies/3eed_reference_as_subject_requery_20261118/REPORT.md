# R169 Querying the reference as the primary referent

**Decision.** Keep R158 plus PCA as the deployment readout. Advance relation-preserving reference requery as a candidate for independent confirmation. Retain reference-phrase requery as a control.

The fixed-bank replay covers 18,620 ordinary directional observations. It exactly reproduces the R168 independent and old left/right-only endpoints. The normalized role-token score with left/right only reaches 14.549% mIoU, 27.421% Acc@25 and 10.136% Acc@50, versus 14.496%, 27.340% and 10.161% for independent selection. Acc@50 reaches 10.063% with the old semantic unary and 10.085% with the permuted semantic correction. The normalized score yields 12 independently image-verified reference selections versus 8 each for the old unary and the permuted correction. This evidence closes the cache-level scoring branch without a deployment change.

The frozen requery experiment comprises the existing 54 verified directional edges from 26 physical frames. Forty-six edges from 22 frames and 13 recordings support a deterministic singular subject/reference swap with a camera-horizontal relation. A uses the original utterance reference-token channel. B canonicalizes the same edge with the original subject. C grounds the reference phrase directly. D grounds the reference as the subject and reverses the relation, retaining the two endpoint descriptions and protected appearance spans. The two canonical versions omit context outside the selected edge.

| Arm | Natural reference top-1 | Mapped top-1 | Natural / mapped reference coverage |
|---|---:|---:|---:|
| A original | 7/46 | 7/46 | 34 / 34 |
| B canonical original-subject | 6/46 | 7/46 | 35 / 29 |
| C reference phrase | 10/46 | 11/46 | 32 / 28 |
| D reference as subject | 15/46 | 12/46 | 36 / 29 |

Correspondence is determined by distinct shared input points. The mapping uses no slot identity or ground-truth assignment. D's correct natural winner links to the known correct identity in 12 cases and an unowned box in 3. Reweighting among linked candidates changes the selected mapping in additional cases. D exceeds C by five correct natural assignments and one mapped assignment. Frozen candidate ownership requires separate validation from successful requery.

Both C and D pass the prespecified identity screen. Adding left/right only with the same D evidence changes subject Acc@25 counts from 20 to 22 and Acc@50 counts from 13 to 15, with 2 Acc@50 rescues and no harms on the eligible panel. Both rescues have verified subject and reference identities and come from different recordings. C yields 20 to 22 and 13 to 14. Across all 54 attempted edges, retaining the frozen subject when the rewrite is unsupported, D gives 17 to 19 Acc@50 successes. Eighteen new source-only image reviews find eleven proxy matches, four contradicted associations and three unknowns. The source review is descriptive.

Predictions precede evaluation. The three source checkpoints and final candidate geometries are held fixed and no FINAL access occurred. The frozen forwards and the single-edge geometry restriction define the scope of these diagnostic results.

[Full Chinese report](REPORT_ZH.md) · [Design](experiment/PROTOCOL.json) · [Independent scalar audit](SCALAR_AUDIT.json)
