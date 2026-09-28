# FP32 task-head source interface v1

Registered before GPU; engineering interface, no optimization/native/target effect.

Fixed first source28199/5624461612/32frames, exact B1 and saved supervised3. Original model.load BF16, deterministic torch/cudnn/CUBLAS:4096:8, original GT-prefix/context/position, original joint_loss/chunks32/token means/complete152775 vocabulary, original branch_injection. Same observed clean source pixels/query fields. SourceGT only supervised diagnostic. No optimizer, interpolation, changed steps/LR/lambda or target/native prediction.

Only change: within the task CE context the frozen original Linear head computes F.linear(h.float(), weight.float()) without autocast/TF32. Weights are unmodified BF16 storage; the head forward is restored on context exit including exceptions and remains active through checkpoint backward. Body, adapter dtype, original native decode, parameter update scope and objective support unchanged. Temporary full FP32 weight approximately1.564GB, original output projection checkpoint retained, no attention/KV checkpoint change, no support slicing. Native module never left patched.

CPU controls: disabled helper exact original loss/stats/hidden gradient; active full-vocabulary FP32 reference and original NTP/MTP counts/means including multiple32-token chunks; frozen weight and exception restoration. These are synthetic only.

GPU two states x two branches =4 forward/backward pairs. Enabled calibration scope exactly66816; all PTD/head/gates/out/textpool frozen, autograd.grad returned vector only; no parameter.grad accumulation. Capture full FP32 vocabulary outputs/hidden and original task tokens plus raw parameter gradients. Before backward save forward and remove observation hook to avoid checkpoint double counting. Physical identities, hidden hash, injection endpoints, GT support and adapter identities must equal previous source_head_projection/cast evidence. All162 token FP32 targetlogit/logsumexp/CE compared with sealed FP64 reference: predeclared absolute tolerance2e-4, rtol0; original chunk branch CE within2e-4 of reference means. Hidden remains exactBF16, no alternate model precision. Require finite/nonzero gradients, exact B1 reset, scope freeze. Invalid interface retains failure evidence and does not enter16-source training.

Resources: GPU600s engineering limit, storage130,000,000bytes and actual serialized checks with1MB metadata buffer, 8GiB disk floor; prior38324.86620450403s/capnull, all worker/wrapper/failures charged. SingleGPU lease/oneLuna. Current freeGPU31366MiB, disk9.335GB. Independent CPU after seal computes all token differences and gradient norms from raw; no extra GT. CPU max600s/4GiB RSS,4threads. No effect selection or tolerance adjustment after results.

If interface passes, separately register fixed PANEL16 source supervised positive control with only this final projection change; old BF16 controls reused only after matching identities. Report original BF16 and newFP32 CE on same endpoint states separately, actualAdam/delta/native geometry, fixed3steps. Interface success is not training success. If fails, inspect exact support/precision/memory and version repair, never skip examples.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
