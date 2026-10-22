# Matched low-resolution self-hull controls

Explicit posthoc mechanism control after primary63 and high-self-hull scores; not preregistered main result or independent confirmation. No sweep, fixed offset0 primary.

| Method | tIoU % | Full vIoU % | Mean interval s |
|---|---:|---:|---:|
| Only224six | 56.2981 | 30.0946 | 8.6358 |
| Only448 | 57.7307 | 31.4231 | 8.6144 |
| Fusion | 59.1483 | 31.8466 | 9.0850 |
| SelfFusion0 | 58.2852 | 31.2180 | 9.0922 |
| SelfFusion1 | 59.0460 | 31.7769 | 8.9160 |
| LowSelf0 | 53.3870 | 28.5013 | 8.1536 |
| LowSelf1 | 54.2125 | 28.6848 | 8.1837 |
| LowSelfFusion0 | 55.7175 | 29.7569 | 8.6184 |
| LowSelfFusion1 | 55.0693 | 29.3696 | 8.3753 |
| LocalSelfFusion0 | 55.6499 | 29.8829 | 8.6982 |
| ShiftedSelfFusion0 | 55.4355 | 29.6156 | 8.6530 |

Six low224/local cached arms, no new image/text/head/decoder/expert. Independent draws of the same view are enveloped by the ORIGINAL hull operator. This widens a posterior without adding second-view visual evidence. No per-query branch selection; all signed losses and fixed-space effects are retained.
