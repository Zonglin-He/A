# R1 semantic preference first: schedule-only amendment002

2026-10-17 explicit user: 可以先做语义偏好的r1。
The previously fixed8-parent R1 half is executed now, independent of Qwen download
or scientific publication. Same HC4/Vid4, slot0, same original E1 first4 legal
candidates, same labelled full anchor/context, same prompt, single-token posterior,
both label orders; max16 R1 native forward calls, zero generation/Student/GT.
No new candidate/Prompt/optimizer choice from outcomes.

Only the Qwen CPU controller is held while R1 owns the GPU; original download is
untouched. Before SIGSTOP, main Qwen status was waiting_download, no science
ACTIVE_EXECUTION/claims existed, and its cgroup contained only the controller PID.
Qwen PID+start ticks+unit+cgroup identity are bound. R1 normal/failure finalizer
and ExecStopPost release that exact CPU PID; the Qwen resource-admission gate
still prevents loading while insufficient GPU memory is free. No GPU worker is
signalled and no second download or model queue is created.

Original preference interface001 CPU waiter is retired, with runtime/receipts
preserved. New early unit stvg-semantic-preference-r1-002.service uses collector
v2/runtime002; the sole dependency change applies to R1 only. Later preference
interface002 waits for the main Qwen32 publication, reuses exact sealed R1 outputs,
then runs only the Qwen half and joint root. It cannot repeat R1 calls. Total
interface compute remains7200s: later Qwen gets the remainder after actual R1
worker wall, dependency waiting excluded. Original dependency deadline unchanged.

R1_ROOT_READBACK pass8 is engineering evidence: correct token IDs, finite model
probabilities, full-vocabulary allowed-label mass, permutation restoration and
exact output hashes. It is not semantic correctness, a Student result or the full
2x2 mechanism experiment. Publish the finite R1 interface independently, including
abstentions and label-order sensitivity, after scalar/pixel/privacy/remote checks.
Next root must lock the matched geometric/preference Student support/gates and
response conversion before any fit. Qwen replacement and failed confirmation
remain separate; CURRENT/paper unchanged.
