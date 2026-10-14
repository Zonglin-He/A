# P0-1 / P0-2 protocol record

This is a de-identified public protocol record of the finite run, transcribed from the report implementation and sealed anonymous outputs. It preserves the run design while omitting raw stimuli, query text, and internal path-bearing lock files.

## P0-1: candidate representation controls

- Temporal comparison: paired sparse snippets versus two contiguous local boundary windows sampled uniformly at stride four.
- Spatial comparison: the same eight physical frames, paired small panels versus independent 448-square full-scene views. Evidence and pixel budgets differ, so this does not isolate resolution as a single cause.
- Each domain/branch/representation cell contains four fixed paired comparisons. ID/order consistency, stable preference, and informative correctness use their separately reported denominators. Tie and Uncertain are not update directions.

## P0-2: query-conditioned task controls

- Native-style output adapts the official Target ID / Time range syntax to supplied hypotheses. It is not open-world benchmark accuracy.
- Simplified alignment receives partial query-free subject/action facts plus the same visuals and the query. Boundaries are not judged without the query.
- The coarse control truncates a student box by 55% on the right. Neither candidate is predeclared correct; local candidates are not separate object instances.

The task used eight historically exposed development queries. There was no fresh confirmation cohort; training overlap is unknown. The saved post-seal root receipt records ground-truth access for scoring.
