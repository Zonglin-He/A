# R76 source-prepared geometry-constrained bidirectional fusion

User now authorizes Waymo paired-source preparation of new RGB/interaction modules, original3EED frozen. Hypothesis: physically local sequential LiDAR→RGB→LiDAR attention plus supervised source grounding aligns self-supervised physical recovery with target selection, unlike R70 cold RGB/R74-R75 unprepared auxiliaries.

Fixed512WaymoTRAIN/128WaymoVAL recordingdisjoint,8epochs3armsseed0,sourceVALonly checkpoint selection. NewrandomCNN, noexperts/pretrainedvision. OriginalA256/sourcequeryanchors/positionsfixed, sourcebasefrozen. Depthheadstrictimageonlypreinteraction; complementaryRGBpixel/pointfeaturemask stops directtargetcopy. TargetDEV336 frozen/control/TTA1/3 predictions before task labels. Unknowns/negatives retained, noFINAL. Noautoexpansion.

Before any target GT: source attempt001 stopped after discovering late RGB linear bias reached unknown-anchor candidates. All partial states/curves retained, no target performance selection. Fix multiplies projected RGB delta by calibrated support mask. Source attempt002 repeats unchanged seed/data/8epochs/optimizer; captures and physical graphs reused.
