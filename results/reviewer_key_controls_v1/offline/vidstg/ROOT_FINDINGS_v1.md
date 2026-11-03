# Root findings: VidSTG spatial controls

The full-domain independent offline audit passed. The 2-4s and >4s gains survive restriction to the exact current predicted event interval: +3.771976pp [2.756186,4.817705] and +3.848319pp [2.460074,5.371713], on1987queries/22785annotatedframes and761queries/11722frames, respectively. This weakens the explanation that long-distance benefits only occur outside the final interval.

At0-.5s INSIDE the predicted interval, the path is worse than original interpolation: -1.236236pp [-1.691648,-.781338],7650queries/75095frames. Across all distances in that interval, the query-equal mean is +.257124pp [-.238486,.742046], not a clear whole-support advantage. This descriptive reweighting does not uniquely decompose dense tube vIoU.

Fixed expert-quality thirds do not support the proposed simple low-quality-expert explanation: high-quality anchors have Ours-minus-interpolation vIoU -.599935pp [-.910924,-.299904],3450queries; declarative high -.508097pp [-.841980,-.178508],2423queries; interrogative high -.816607pp [-1.225773,-.414596],1027queries. Lowest-third whole spatial difference +.155186pp [.012837,.287952] is pointwise and conditional, not a quality gate. Missing quality1796queries remains separate. Strong interpolation when anchors are accurate is a plausible interpretation, not a measured unique cause.

All near/far, event-membership, quality and query-form groups retained, including empty cells. Query equal and parent-cluster10000pointwise95%CIs; frames not independent. Full sets previously exposed; no default change, online GT use or paper edit.
