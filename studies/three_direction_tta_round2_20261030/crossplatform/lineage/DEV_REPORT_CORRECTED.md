# Corrected round2 fixed18 development evidence

This supersedes task scores and their consequences in `DEV_REPORT_V1.md`. Original predictions/teacher/features and invalid score/report files remain preserved. The oldscorer passed float32yaws into NumPytrigonometry; the resulting slightlynonorthogonal matrices violated strict1e-9OBBhalfspace tolerances. New `score_sealed_double.py` casts yaw toPythonfloat and checks float64orthogonality before scoring. Fourteenof144IoUs change. Sourceyaw0 and parameter/geometry diagnostics are unaffected.

| Arm | Corrected mIoU% | Acc>.25 | Acc>.50 |
|---|---:|---:|---:|
| Source |30.9823|16/18|1/18|
| Fixed PCA |51.3387|16/18|10/18|
| Fixed surface |50.3332|16/18|10/18|
| Directpartialface latent |30.9816|16/18|1/18|
| Adapter0 |30.9823|16/18|1/18|
| Adapter1 |28.3710|15/18|1/18|
| Adapter3 |28.2827|16/18|1/18|
| Source-targetcontrol3 |30.9823|16/18|1/18|

The apparent zero-IoU tails in the firstreport were measurementerrors and are retracted. Onlytwoofthethreeupdatedframes deteriorate; oneimproves: source.399474→adapter1.623992→adapter3.557544. Negativeexamples remain real: source.786879→adapter3.251723, andsource.454691→.345850. Allpositive/negative tails are retained in `dev_scored_double.json`. The directlatent is essentiallysource; PCA/surface are strongcontrols and currently outperform the structuredadapter.

Measured gatecoverage is still3/6Waymo and0/6eachDrone/Quad,10acceptedcandidates. Noidentityswitch occurs. SharedWspillover is stillreal, including changes to unaccepted chosenqueries. The label-free loss rises afterthefirstAdamstep onall3updatedframes, andRMSfeatureL1norm220–235 stillprovides an optimizerconditioning explanation. These observations justify testing optimizerrepair without claiming that everyposechange or everyacceptedteacher is incorrect.

`V2_PROTOCOL_DRAFT.md` was a draftonly, neverexecuted. Its selection and finalfreeze must use thesecorrectednumbers, not the invalidzeros. Confirmation36 remainsuninferred/unscored. No further scientificallytunedthresholds have been executed.

Finite staticreview of previousround1geometry scorers found their prediction arrays explicitly castwith `.astype(float)` beforeyawtrigonometry. Rootindependent geometry/surface auditors also used SciPyfloat64Rotation andhad~5e-8maximumerrors. The float32rotationbug was introduced by the newround2scorer and doesnot invalidate thoseearlierresults. See `ROUND1_SCORER_PRECISION_REVIEW.json`; no oldfiles or scores were modified.
