# Same-offset hull: posthoc cached control

Explicit posthoc mechanism control after primary63 scores; not preregistered main result or independent confirmation. No sweep, fixed offset0 primary.

| Method | tIoU % | Full vIoU % | Mean interval s |
|---|---:|---:|---:|
| Only224six | 56.2981 | 30.0946 | 8.6358 |
| Only448 | 57.7307 | 31.4231 | 8.6144 |
| Fusion | 59.1483 | 31.8466 | 9.0850 |
| SingleHigh0 | 51.2004 | 28.5437 | 6.8269 |
| SingleHigh1 | 50.1973 | 27.3835 | 6.6744 |
| SingleFusion0 | 56.3604 | 30.4439 | 8.4346 |
| SingleFusion1 | 56.9716 | 30.6157 | 8.3008 |
| SelfHull0 | 56.8942 | 30.9962 | 8.3323 |
| SelfHull1 | 56.2220 | 30.5706 | 8.1239 |
| SelfFusion0 | 58.2852 | 31.2180 | 9.0922 |
| SelfFusion1 | 59.0460 | 31.7769 | 8.9160 |

Four new cached arms, no new image/text/head/decoder/expert. Independent draws of the same view are enveloped by the ORIGINAL hull operator. This widens a posterior without adding second-view visual evidence. No per-query branch selection; all signed losses and fixed-space effects are retained.
