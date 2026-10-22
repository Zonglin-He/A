# Preserve global appearance; focus target-context motion

Exposed HC64 prospective fixed branch. Earlier both-modality fill/restoration cannot isolate loss of actor/interaction appearance from competing motion. New main retains ORIGINAL 2D appearance everywhere; original 3D motion retained inside fixed1.5 fulltime complete-box context, outsidevalid filled with sameframe inside3D mean FP32thenFP16. Global actors/objects can remain in appearance. Original motion has fullclipcontext; no claim of pure local action isolation.

Original complete caption, all65 physicalnodes, original dense keys/padding/safetycorner, native fusion/two passes/six layers/two-offset hull/physical MBR and final complete-boxDP/eight observations unchanged. EmptyROIframe reverts oldfullvalid, recorded/noqueryskip. Reflected3D samearea, appearanceonly andreflected2D role controls, wrong-time3D. Wrong-time reverses original featureframeorder within each offset beforecurrentROI/fill: clips NOT internally reversed, not a true backwardvideo or knownfalseevent. Primary fixed halfGlobal6+halfMotionTarget; DirectMotion sameinformation. BothFocus EXACT old17FillOnlyposterior/interval/finalbox alias with0newmodelcalls, not renamed newmethod.

All64 active; all64 predictions seal before independent CPU official labels. Source/Native/self/noop aliases, Full64hidden/logits/q/Sourceboxes bitwise. No new RGB/visualencoder/text/expert/observations/querylearning/optimizer; no context/fusion/modality-blend strengthscan. FourCPUtests modalitypreservation/insidepadding/reflectionfallback/temporalindexroundtrip.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|MotionFocusFusion|29.4787|55.2220|17/25/22|8|
|DirectMotionTarget|28.4359|52.2051|20/33/11|19|
|MotionReflectedFusion|29.6681|56.0083|24/24/16|12|
|AppearanceFocusFusion|30.3780|56.4480|31/24/9|12|
|AppearanceReflectedFusion|29.3765|54.8086|30/29/5|22|
|WrongTimeMotionFusion|29.1912|53.6186|26/32/6|21|
|BothFocusFusion|30.1406|56.5246|31/25/8|14|
|NativeSelfMixture|29.6952|55.9613|0/0/64|0|
|NoView|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−Global6 t-0.7394pp CI[-1.878550435607135, 0.2605289075862857]; v-0.2165pp CI[-0.7623908902313967, 0.23332468880212684].
Primary−Source t+0.3990pp; v+2.5849pp. Source short-0.4908/long+0.6258tpp.
Primary−MotionReflectedFusion t-0.7863pp CI[-2.5702381640286918, 0.9157301433057589].
Primary−AppearanceFocusFusion t-1.2261pp CI[-4.059170205106041, 1.9507653012922406].
Primary−AppearanceReflectedFusion t+0.4133pp CI[-2.3431197036198013, 2.9342929338159713].
Primary−WrongTimeMotionFusion t+1.6033pp CI[-1.5827780657474044, 5.037553765794667].
Primary−BothFocusFusion t-1.3027pp CI[-4.485197787421097, 2.0708751719180114].

Not qualified. No modalityblend/context/fusionweightscan or posthoccontrolpromotion.
Worker81.80s includes loading/researchrecording; calls{'fusion_offsets': 768, 'native_replay_queries': 384, 'new_text_encoder': 0, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_optimizer': 0}. Study sixreads/query not deploymenttiming. Deployment would add one actual sourcefusion/native-time read, but end-to-end cost not measured here.
Official/dense checks1408 maxerror9.99e-16. [All signed results](signed.tsv), [severe tails](severe_declines.tsv), [Source duration groups](source_duration_groups.json), [protocol](protocol.json).

Original [TA-STVG paper section3.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/9b5b12ea5acdf5d79544b412ac32cc51-Paper-Conference.pdf) describes separate appearance/motion features and trained joint fusion; it does not validate this frozen focus change.

Concrete postseal modality comparisons: RC042 badBothFocus−63.16tpp andAppearanceFocus−64.90, whileMotionFocus0vsG6. RC015 BothFocus−24.68/AppearanceFocus−22.47/MotionFocus0. Thus globalappearance preserves baseline in some harms, but does not supply new corrections. RC020 beneficialBothFocus+18.97/AppearanceFocus+21.72 vsMotionFocus0; RC060 BothFocus/AppearanceFocus+22.61 vsMotionFocus0. RC041 harmfulBothFocus−26.75 becomesAppearanceFocus+.93/MotionFocus0. RC011 allrealmodalityfocus−14.04. RC023 realMotionFocus+9.92 butreflectedMotion+15.73; wrongtimeMotion+9.92. RC064 MotionFocus0 butwrongtimeMotion+24.67. Correct-location motion is not established as the source of event-boundary gain. AppearanceFocus controlSource short−.9110/long+2.2715,12severe; cannotpromoteit. Neither a uniform modality split nor stronger refusal solves current feedback quality.
