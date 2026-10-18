# Source validation auxiliary readout

Before training or source validation, record the same initial-E two-level boundary-distribution mean with widths allowed and planned nodes fixed E, as an auxiliary source readout. No extra forward; checkpoint selection and the prelocked center-only direct2 gate are unchanged. This separates head capability from the center-only candidate readout if source qualification fails. Not a new target arm.

Direct2 chooses the largest raw expected-IoU reward over the current center-only candidates. Distribution-distillation arms retain the locked Gaussian sampling-prior tilt; raw and posterior E-relative rankings must therefore be reported separately. This comparison does not isolate an update effect from prior/readout differences. Source validation constructor also retains its temporary external-state readouts; they do not update head weights on validation data or change direct2 checkpoint selection.
