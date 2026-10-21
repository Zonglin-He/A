# Outcome and mechanism boundary

The cached sequential policy has been implemented and evaluated on all 64 prescribed HC2 development parents. Computation and direct unobserved-frame control now work. This configuration has **not established an OPD advantage** over fixed-state distillation or full-gradient Direct, and it does not replace the registered geometric method.

## What actually ran

Each query independently reset six matching weights. OPD and Direct each generated six training paths (two per round), performed three actual Adam updates/backwards, and returned one greedy path from the updated policy. FixedKD generated two teacher paths once and performed six cached-state loss passes in three updates. FixedPropagation returned a greedy teacher-feedback path with no updates. All used the same evidence and candidate bank, with no heavy-network, decoder, expert or RGB calls during this experiment. Direct differentiated through the full soft memory, previous-choice association and teacher target; it was not deliberately deprived of available gradients.

All64 paths, candidates, predictions and receipts were sealed before independent scoring. The official and independent dense metrics agreed in 448 checks to at most 9.99e-16. Original paused formal state and registered method configuration hashes remained unchanged. The 128-parent panel was already development-exposed; selecting parameters on it is disclosed development use, not online GT access. This64 run is not independent confirmation or a complete official-split evaluation.

## Results

OPD's full vIoU is31.2165%, or +0.1248pp over ERef+Strong, with paired95% interval[-0.5233,+0.8400]. Its unobserved-node sIoU is51.5249%, or +0.8194pp, interval[-0.3242,+2.3955]. Both intervals span zero. OPD is -0.0551pp below FixedKD and -0.0661pp below Direct on full vIoU; neither comparison establishes a positive OPD increment. It also remains -0.1702pp below the saved geometric OPD reference.

Relative to ERef, OPD improves10 queries, declines15, and leaves39 unchanged in vIoU. Positive contributions average +0.6462pp across the full panel; negative contributions average -0.5214pp. The net +0.1248pp should not be described as broad or stable improvement.

Every new method has one vIoU decline greater than5pp relative to ERef. The common case H035 loses10.4538pp under OPD, Direct and FixedPropagation, and10.5696pp under FixedKD. It is already badly changed by the no-update propagation policy, so its failure cannot be assigned specifically to the OPD optimizer. H035 has only one of27 scored unobserved nodes with a candidate more than5pp better than E; its average available node-IoU improvement is only0.2458pp. This supports preserving E for this case, but that GT conclusion is offline and was never used for fallback. It does not by itself prove which object's identity the teacher selected.

## Did unobserved-frame control and on-policy execution occur?

Yes. All64 queries changed their temporary matching weights. Final OPD output changes27 query trajectories and158 unobserved nodes relative to E. Unsupported nodes remain exactly E, select action0, and do not write memory. A total1003/3648 unobserved physical nodes are active under the prelocked event/confidence/support gate.

The program did collect current student histories: OPD samples six hard histories per query, with2558 non-E active actions out of7734 active state visits across the panel. These are not teacher-only trajectories or best-path selection. However, final OPD and FixedKD differ on only8 query outputs and20 physical nodes (19 unobserved). Their vIoU differs on only4 queries. Thus this implementation created state dependence without establishing a useful independent benefit from refreshing those states.

## Does the candidate bank contain useful alternatives?

The offline candidate diagnostic finds189/1365 fixed unobserved GT nodes (13.8462%) with a valid candidate more than5pp better than E, distributed across45/64 queries. GT nodewise candidate readout yields32.9824% full vIoU, +1.8906pp over E. This is a diagnostic that keeps observed and all other nodes at E; it is neither an executable method nor an exact upper bound over all possible full trajectories. It shows some available correction, but does not establish the requested2–3pp attainable gain or that these corrections can be identified without GT.

## A structural limitation of this first policy

With features [memory cosine, anchor cosine, projection, incoming link, negative guard, E indicator], the teacher distribution is exactly representable by fixed weights

    theta_E = [0, 0.1, 1, 0.1, 0.1, 0].

The saved-path audit checks this on every recorded training state: maximum conditional log-probability difference is7.11e-15. FixedPropagation is therefore a strong explicit teacher-policy baseline. This construction does not require OPD to make the target representable or usable; three-step fitting differences reflect finite adaptation and state weighting. It would be incorrect to claim that Direct cannot express the teacher or that the sequence formulation alone proves OPD necessary.

OPD uses supervised semi-gradients at hard student-visited states. Direct uses the full pathwise gradient of a soft recurrent relaxation. They do not compute the exact same hard-history expected gradient. OPD vs FixedKD also changes unique state coverage as well as student/teacher behavior and refreshing. These distinctions constrain attribution even if a future configuration shows a numerical difference.

The cached features are Source post-fusion visual tokens, not a trained tracker. Cycle and forward/backward agreement do not certify identity, and the same ReferDINO can give consistently wrong masks and boxes. Moving adaptation to the output side solves the old decoder cost and permits direct changes on unobserved nodes, but it does not create an independent signal that distinguishes a true target correction from a plausible wrong propagation.

## Cost and decision

Mean CPU costs per query: fixed evidence0.0984s, candidate bank0.0335s, OPD adaptation plus final readout0.0621s, FixedKD0.0195s, Direct0.0674s, fixed propagation0.0054s. Shared evidence+bank+OPD is about0.1939s, excluding inherited Source/Strong/ReferDINO generation and file/validation overhead. The64-query run stage covering all four new outputs and sealing took29.01s. The four methods' query-loop total averages0.3379s. These are cache-based incremental timings, not complete deployed inference latency.

The current decision is to retain this as a completed negative/mixed HC64 pilot. No follow-up parameter sweep, new expert, resumed formal task or replacement of the registered method is inferred from these results. All signed outcomes, traces and source identities are preserved; the public export excludes private tensors and identifiers.
