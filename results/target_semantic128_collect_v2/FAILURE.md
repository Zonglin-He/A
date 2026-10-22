# Preserved compact-geometry adapter failure

v2 preserves/migrates7 prior completed records byte-for-byte. The first new Source-only
read reproduced saved Source/all-layer logits, but the adapter stopped before
accepting a new member: compact conversion required the missing native interval
field. Original code/plan/logs/outputs are preserved, with0 new expert calls and no GT.
A separate v3 plan derives the interval from the newly restored original final-layer
native logits, using original native probabilities/physical MBR. Original official
Source/E/Full/interval/eight-position equivalence remains mandatory. This is an
engineering repair with explicit migration, not parameter tuning or a model negative.
