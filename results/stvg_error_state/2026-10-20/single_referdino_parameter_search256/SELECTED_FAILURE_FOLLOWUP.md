# Final configurations: harmful-parent follow-through

Parameters are chosen once per dataset on all128 exposed parents. The following retained baseline-harm group is for offline explanation, not a per-video choice. Original and selected initialization are reported separately. All values are complete vIoU percentage points.

## hc2

| Alias | old E | old Full | selected E | selected Full | selected−old Full | selected OPD−own E |
|---|---:|---:|---:|---:|---:|---:|
| RC036 | +42.92954 | +39.43690 | +44.93833 | +41.02026 | +1.58336 | -3.91807 |
| RC097 | +19.34122 | +16.89670 | +19.64479 | +21.45688 | +4.56018 | +1.81209 |
| RC122 | +51.29036 | +49.66397 | +46.74133 | +46.06658 | -3.59738 | -0.67475 |
| RC125 | +19.76102 | +18.67642 | +23.59873 | +25.24175 | +6.56532 | +1.64301 |
| RC063 | +31.23412 | +30.29653 | +30.28587 | +28.98292 | -1.31361 | -1.30294 |
| RC080 | +32.98913 | +32.13614 | +40.89333 | +39.19493 | +7.05880 | -1.69840 |
| RC109 | +52.31355 | +51.49653 | +55.77273 | +54.20604 | +2.70951 | -1.56670 |
| RC070 | +3.31197 | +2.80254 | +8.87289 | +7.72117 | +4.91863 | -1.15172 |
| RC114 | +34.49324 | +34.01128 | +15.21105 | +15.85782 | -18.15346 | +0.64676 |
| RC025 | +32.32577 | +31.88163 | +33.65309 | +34.23383 | +2.35219 | +0.58074 |
| RC088 | +20.31958 | +19.91926 | +17.25958 | +17.22417 | -2.69509 | -0.03541 |
| RC060 | +43.33824 | +43.10840 | +43.59537 | +43.29814 | +0.18973 | -0.29723 |

All-parent, original-harm and remaining-parent aggregates: `{'all_parents': {'parents': 128, 'mean_gain_vs_original_Full_pp': 0.7601440543842862, 'improved_vs_original': 62, 'worse_vs_original': 63, 'original_mean_adaptation_delta_pp': 0.2993980792583869, 'selected_mean_adaptation_delta_pp': 0.4258627218652088, 'selected_severe_relative_own_E': 0}, 'original_harmed_group': {'parents': 54, 'mean_gain_vs_original_Full_pp': 1.7726962272837883, 'improved_vs_original': 30, 'worse_vs_original': 24, 'original_mean_adaptation_delta_pp': -0.2867763615627986, 'selected_mean_adaptation_delta_pp': -0.10536732595260404, 'selected_severe_relative_own_E': 0}, 'remaining_group': {'parents': 74, 'mean_gain_vs_original_Full_pp': 0.02125463091708233, 'improved_vs_original': 32, 'worse_vs_original': 39, 'original_mean_adaptation_delta_pp': 0.7271469955333061, 'selected_mean_adaptation_delta_pp': 0.8135170810836126, 'selected_severe_relative_own_E': 0}}`.

## vidstg

| Alias | old E | old Full | selected E | selected Full | selected−old Full | selected OPD−own E |
|---|---:|---:|---:|---:|---:|---:|
| BV030 | +10.84186 | +3.68347 | +10.84186 | +3.72549 | +0.04202 | -7.11637 |
| BV022 | +9.66203 | +5.12764 | +9.66203 | +7.84377 | +2.71614 | -1.81826 |
| BV064 | +26.01834 | +23.21802 | +26.01834 | +23.14171 | -0.07631 | -2.87663 |
| BV083 | +4.31988 | +2.67242 | +4.31988 | +2.68860 | +0.01617 | -1.63128 |
| BV023 | +20.27929 | +18.83677 | +20.27929 | +20.77769 | +1.94093 | +0.49840 |
| BV114 | +9.59502 | +8.20208 | +9.59502 | +7.92681 | -0.27527 | -1.66821 |
| BV108 | +44.33942 | +43.27132 | +44.33942 | +41.42174 | -1.84958 | -2.91768 |
| BV074 | +3.62538 | +2.86137 | +3.62538 | +3.11814 | +0.25677 | -0.50724 |
| BV047 | +51.36056 | +50.78083 | +51.36056 | +50.77006 | -0.01078 | -0.59050 |
| BV098 | +5.25062 | +4.70077 | +5.25062 | +4.80153 | +0.10076 | -0.44909 |
| BV103 | +59.27198 | +58.83797 | +59.27198 | +59.03971 | +0.20174 | -0.23227 |
| BV072 | +9.61702 | +9.19421 | +9.61702 | +9.26284 | +0.06863 | -0.35418 |
| BV041 | +7.28047 | +7.24872 | +7.28047 | +7.24905 | +0.00033 | -0.03142 |

All-parent, original-harm and remaining-parent aggregates: `{'all_parents': {'parents': 128, 'mean_gain_vs_original_Full_pp': 0.03577801605175181, 'improved_vs_original': 53, 'worse_vs_original': 46, 'original_mean_adaptation_delta_pp': 0.06507866954234118, 'selected_mean_adaptation_delta_pp': 0.10085668559409303, 'selected_severe_relative_own_E': 1}, 'original_harmed_group': {'parents': 39, 'mean_gain_vs_original_Full_pp': 0.0969369915304209, 'improved_vs_original': 20, 'worse_vs_original': 19, 'original_mean_adaptation_delta_pp': -0.6439697779783916, 'selected_mean_adaptation_delta_pp': -0.5470327864479707, 'selected_severe_relative_own_E': 1}, 'remaining_group': {'parents': 89, 'mean_gain_vs_original_Full_pp': 0.008978015561099078, 'improved_vs_original': 33, 'worse_vs_original': 27, 'original_mean_adaptation_delta_pp': 0.37578529261322413, 'selected_mean_adaptation_delta_pp': 0.38476330817432314, 'selected_severe_relative_own_E': 0}}`.

