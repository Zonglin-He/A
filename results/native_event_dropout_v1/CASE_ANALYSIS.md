# Independent postseal dropout cases

All stochastic modules actually use p=.1. No native source change. Gaussian interval equality is decision-level equality, not equal posterior or disabled dropout.

HC64
|Alias|Δt G6 pp|Δv G6 pp|boundary Δ seconds|
|---|---:|---:|---:|
|RC014|-3.8829|-2.5877|+0.3194|
|RC033|-2.1277|-0.0601|+0.1667|
|RC001|-2.0764|-0.2708|+0.2994|
|RC020|+2.7586|+2.5444|-0.1600|
|RC004|+3.3981|+1.2198|-0.1400|
|RC034|+7.5377|+2.4720|-0.2994|
All128 Gaussian-moment control final intervals match primary; the saved posteriors differ. These cases do not support event-specific stochastic correction.

hc2_recheck
|Alias|Δt G6 pp|Δv G6 pp|boundary Δ seconds|
|---|---:|---:|---:|
|HC2_RECHECK017|-7.6306|-6.1881|+0.3194|
|HC2_RECHECK031|-2.6217|-1.1123|+0.1400|
|HC2_RECHECK003|-2.3565|-1.4114|+0.1498|
|HC2_RECHECK023|+4.0000|+3.1117|-0.1597|
|HC2_RECHECK012|+4.0598|+3.5178|-0.3161|
|HC2_RECHECK018|+14.2857|+9.6346|-0.6156|
All128 Gaussian-moment control final intervals match primary; the saved posteriors differ. These cases do not support event-specific stochastic correction.

vidstg_recheck
|Alias|Δt G6 pp|Δv G6 pp|boundary Δ seconds|
|---|---:|---:|---:|
|VIDSTG_RECHECK028|-1.8384|-1.4223|+0.1001|
|VIDSTG_RECHECK021|-0.7812|-0.4433|+0.2002|
|VIDSTG_RECHECK024|-0.6595|-0.5955|+0.1875|
|VIDSTG_RECHECK009|+1.4168|+1.0704|-0.4004|
|VIDSTG_RECHECK004|+4.0323|+3.8511|-0.1000|
|VIDSTG_RECHECK020|+4.9587|+2.1886|-0.1001|
All128 Gaussian-moment control final intervals match primary; the saved posteriors differ. These cases do not support event-specific stochastic correction.
