# Data and publication boundaries

Inference accepts video, caption and predetermined physical input support. No target
boxes/event interval/GT-dependent choice enters ERef, teacher inference, propagation,
time readout or final decoding. Source construction replaces the two upstream annotation
loads with empty mappings. The guard checks surface paths and resolved symlinks; it is
an engineering boundary, not a sandbox for arbitrary Python.

The scorer validates the complete prediction seal and hashes before reading labels.
Synthetic engineering tests cover rejection paths. Cleanup does not read real labels
or score new predictions. Saved-capture compatibility reads only existing model outputs.

HC2 official 1902/227 and VidSTG 10303/732 lists use the corrected official membership
and endpoint conversion. Historical raw3482 and older metric reports are archived with
their original provenance; they must not be mixed with the current full protocol.
Development panels have exposure and are not independent tests. Full target evaluation
cannot be used to retune the frozen profile. Historical exposure and corrections are
available in the research archive at c310af5888dca78d6ea7b0a280cb67d209a90337.

The public tree excludes videos, raw requests/query text, annotations, weights, raw
masks/features/states/caches, local links, environments and credentials. An explicit
source list is exported and remote file bytes verified. Private assets and all old
saved predictions remain local. No claim about absent upstream pretraining contamination
follows from these local data-flow checks.
