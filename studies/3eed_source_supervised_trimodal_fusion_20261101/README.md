# R25 — source-supervised RGB/point/text fusion

A real candidate-level trimodal source model: frozen Waymo-only point/text backbone and boxes, frozen ImageNet ResNet34 layer2/3, calibrated local RGB ROI tokens, language-conditioned cross-attention, learned feature gate and matching/rank head. No Qwen in model training or inference. This is source-model construction, NOT a TTA result (zero target updates).

Waymo TRAIN3682, sourceVAL512, two finite readouts × RGB/noRGB × two fixed seeds,30epochs each. All source curves retained. SourceVAL chooses residual-rank family before any target scores;136930new parameters, original matching projection frozen. Point backbone was not retrained.

Target400 expressions (200/platform), including64 new current-input content-disjoint observations,2Drone/3Quadclusters; allVAL historically scored. Main seed mean mIoU Source22.566555%, noRGB22.512782%, RGB23.025109%. RGB vs matched noRGB +0.512327pp (second seed +0.106908). On new64, RGB vs noRGB +1.792572/+0.903738pp, but vs originalSource -0.641229/-0.005278pp. Quad decline is retained; source-trained noRGB ranking also harms that split. RGB benefit is small and platform/seed dependent. Keep research source prototype, not replacement for the stronger Qwen precision baseline and not robust cross-platform superiority.

Both source-readout families were predicted/sealed before current target scoring; no target-driven winner switch. RGB-zero, within-observation query/RGB permutation, global-only, source-supervised noRGB and previousQwen old336 control retained. 24raw-input complete-model closure probes reproduce cached scores and bitexact boxes without inference GT/cache dependency.

An early R25 source-cache coordinate bug omitted official Waymo view rotation, detected BEFORE any fusion training. Invalid001 caches/labels/partialRGB and code are retained and clearly excluded; corrected002 caches used throughout. Earlier R24 target experiments/source checkpoint unaffected. All5source camera views audited; calibration negative tails retained.

13232anonymous scalar rows cover both target families, selected source checkpoints and oldQwen336; they reuse4594unique current inputs, not13232independent samples. All56source curves and421negative comparison records retained. Own code/protocol/scalars only; no raw captions/media/pointclouds/calibration coordinates/task IDs/features/model weights or third-party implementation.

Read REPORT.md and results/CPU_AUDIT.json, DECISION.json, protocol/*LOCK.json. Prior R24 dependency commit86ad9bbbe0b9c6ddb0dabfdd443f52f8b96f8466 and earlier licensed data/source assets are required. Valid workflow: select25; corrected capture25; rgb25; source_labels25; train25 and train_rank25; select_model25; evaluate25 predict; infer25 closure; evaluate_alternative25 predict; score both; audit25; render25; finalize25. Repair/attempt001 scripts are historical engineering evidence, not steps to rerun over valid outputs.
