# ROOT pilot comparison: Source-prompt SAM2.1 + Source-box TR

|Method|HC tIoU|HC vIoU|Vid tIoU|Vid vIoU|
|---|---:|---:|---:|---:|
|Source|46.509478|25.124253|32.121535|13.322903|
|SourceTR|54.833053|30.060976|35.528603|14.548333|
|SAM2Native|46.509478|25.383772|32.121535|15.808445|
|SAM2TR|54.833053|29.917082|35.528603|17.032514|
|Current|54.968548|31.791298|35.202435|15.339259|

Full-SAM2TR:
hc2: +1.874217 pp,95%CI [-0.7423495170406896, 4.477903517688564].
vidstg: -1.693255 pp,95%CI [-3.9636379878485513, 0.521166022542245].

The HC mean favors Current Full and Vid mean favors SAM2TR; both panel vIoU CIs cross zero. No established Full vIoU advantage in both domains. SAM2TR-SourceTR is null in HC and positive in Vid on a genuinely shared SourceTR interval. Primary source-prompt SAM2 is not a same-ReferDINO-evidence propagation control. All negative tails, empty masks, original TTFT results and Source acquisition costs remain.

New stage remains LOCAL until a new verified publication receipt; ongoing full prefixes are not scored or used to select this configuration.
