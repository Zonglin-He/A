# LLaVA-ST both original128 spatial selections

Both domain locks completed before full execution. Same shared merged checkpoint on both domains; no source-only cross-domain claim. Native time identical Source/Ours; frozen spatial numeric grid, all signed positive/negative configurations retained.

|Target|Profiles|Source v|Selected v|delta v pp|delta dense pp|delta unobserved pp|native t|
|---|---:|---:|---:|---:|---:|---:|---:|
|vidstg|20|13.916442|18.490422|+4.573980|+5.921254|+5.603109|45.983889|
|hc2|21|9.563491|12.745927|+3.182436|+5.317257|+5.133143|23.371198|

Vid parameters: gamma1/admission.05/guard.025/link.5. HC parameters: gamma.5/admission.05/guard.1/link.5.

Descriptive paired intervals and tails are in each selection_diagnosis.json and development_report.md. These are selected, repeatedly exposed128 panels; intervals do not correct for selection or training exposure and do not establish independent stable generalization. Full12205 target outputs are pending; do not treat partial inference as complete evaluation.

Execution order after both locks: VidSTG10303 then HC21902. Reuse exactly the selected128 prediction payloads, execute remaining inputs with new backbone. Invalid outputs retained; no full-feedback tuning.
