# Fixed main method

The two directions share the same algorithm. The source checkpoint is trained on the
other dataset. Target inference reads only video, caption and the official input support.

The frozen source runs its original height-224 native two-pass process at two offsets.
Original observation selection supplies eight positions to the same ReferDINO. Its
complete native boxes provide extent; selected soft masks weight source-feature
correspondence. Weighted bidirectional axis-affine fits transport complete boxes to
unobserved frames. Forward and backward hypotheses remain separate, with the initialized
source box always available. Masks are not thresholded into component bounding boxes.
The confidence is return-cycle support, not a probability of correct identity or event.

ERef calibration and gamma remain source-domain calibrated and frozen. Final candidates
are ERef, preceding-anchor propagation and following-anchor propagation. Exact duplicate
hypotheses are pruned in the original order. Fixed reliability admission is 0.1.

For each of six native temporal layers, apply the original frozen time head and original
two-offset hull on the physical frame grid. Average the six normalized interval posterior
tables. Temperature is 1. The final interval uses the original physical MBR. There is
no additional 448 encoding or time-decoder execution.

On that interval, the fixed spatial objective is:

- 0.5 times mean observed-frame support;
- 0.5 times mean unobserved-frame confidence-weighted support;
- 0.25 times mean event-edge confidence-weighted connection;
- minus 0.1 times mean geometric protection cost on all physical nodes.

Denominators are the original plain node/edge counts, not sums of confidences.
The original exact three-state dynamic program chooses the path; ineligible nodes keep
ERef. No optimizer or temporary trainable state is used. Spatial evidence does not alter
the temporal posterior in this selected method.

The cleanup moves the previously script-based Global6 readout into stvg/global6.py.
Native forward, head operations, hidden export, complete-box geometry, score, tie order,
ERef, sampling, precision and input decoding are preserved. Numerical and static scope
is recorded in [verification](../results/verification.json). This is not permission to
reuse old receipts with a new runtime fingerprint.
