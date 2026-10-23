# Preparation/environment failure, not a method result

The first parse-only preparation attempt had an `else732` token error before a
plan/model/prediction existed. Its source and failure record are privately preserved.

The subsequently frozen v1 plan prepared the exact64 additional-parent inputs.
Importing the existing CPU-only helper set `CUDA_VISIBLE_DEVICES=''`; the strict
Source loader rejected CUDA before any Source forward or prediction. No labels
were scored. Original code, plan, logs/failure state were retained, owned PID546753/
start228278565 exited and shared GPU lock was verified free.

V2 has an independent plan/fingerprint and restores the caller's CUDA visibility
before torch initialization. It asserts exact identity of all64 predeclared inputs.
The prototype algorithm, sampling, precision, seed, controls and qualification
tradeoff are unchanged. No old hash is waived; there is no inference-result migration
because v1 produced zero predictions. K16 full remains paused.
