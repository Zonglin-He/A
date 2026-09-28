# Direction audit v2 schema recovery

v1 stopped on CPU before direction results: early actuation time_distribution contains endpoint_logits only; later records additionally contain scope/probe_count/time_token_ids. All actual endpoint logits and remaining native fields were exactly equal in root readback. Original script, lock and failure remain.

v2 only compares shared semantic fields explicitly, preserving exact endpoint logits plus all reference/coordinate/geometry/preprocess fields, and validates extra metadata separately (two probes, unique time support of observed length). Original inputs, objectives, two cases, all directions, numerical tolerance and conditional gate remain as v1. Isolated output direction_alignment_v2; CPU only and zero GPU.
