# hc2: standalone retained192 gamma review control

Exactly192queries, 192 parents. This pre-existing retained-raw cohort includes previously exposed development and confirmation sources; it is not the full1902/10303 domain, an unseen test, or a prefix accuracy estimate. All192 prediction configurations sealed before this independent official scoring. Separate actual ROI repooling/cold gamma chain, prediction/input/parent/receipt SHA and all-mean audits passed.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|44.661062|23.524963|38.541667|13.541667|
|Gamma0|54.186687|29.725522|47.395833|19.791667|
|Gamma025|54.104834|30.114621|47.916667|20.312500|
|Gamma05|54.096368|30.508756|48.437500|21.354167|
|Gamma075|53.925280|30.723044|48.958333|21.875000|
|Gamma1|53.766020|30.474341|48.958333|20.833333|

Parent-paired10000-bootstrap pointwise95% CIs, no familywise significance claim:

- Gamma0-Source: t+9.525625pp CI[6.936603125409502, 12.167594271801939]; v+6.200559pp CI[3.929315905758014, 8.57554982698876]; severe t/v/union29/25/42, rescues107.
- Gamma025-Source: t+9.443772pp CI[6.811851728660036, 12.125251736342475]; v+6.589657pp CI[4.335589303502684, 8.925136211974086]; severe t/v/union29/25/42, rescues109.
- Gamma05-Source: t+9.435307pp CI[6.847394977493687, 12.057960342977392]; v+6.983792pp CI[4.670941002160638, 9.408689551568013]; severe t/v/union28/28/43, rescues115.
- Gamma075-Source: t+9.264218pp CI[6.666337949392151, 11.940075872351736]; v+7.198081pp CI[4.843552104345398, 9.671315110505203]; severe t/v/union32/29/48, rescues116.
- Gamma1-Source: t+9.104958pp CI[6.492654411798231, 11.774159067105916]; v+6.949377pp CI[4.554029053348475, 9.476604644527521]; severe t/v/union33/32/49, rescues117.
- Gamma0-Gamma05: t+0.090319pp CI[-0.411524216089363, 0.5827423440941807]; v-0.783233pp CI[-1.225293031880843, -0.3419340597930615]; severe t/v/union10/13/19, rescues16.
- Gamma025-Gamma05: t+0.008465pp CI[-0.39815711308727586, 0.3823184856659421]; v-0.394135pp CI[-0.7016902327419974, -0.08786957856362836]; severe t/v/union5/8/10, rescues8.
- Gamma075-Gamma05: t-0.171088pp CI[-0.5182776416006394, 0.1372865105575209]; v+0.214288pp CI[-0.05072743556605573, 0.4880434937373974]; severe t/v/union8/2/8, rescues9.
- Gamma1-Gamma05: t-0.330349pp CI[-0.9632702153179815, 0.3063842416313661]; v-0.034415pp CI[-0.5072121413058238, 0.4357496307491973]; severe t/v/union14/9/16, rescues16.

Short<=5s/long ground-truth-duration strata are offline only:

- short_le5s, n59, Gamma0-Source: v+1.769275pp CI[-1.6030034612483102, 5.229229037074871]; t+3.552793 CI[-0.9461580615786108, 8.148523679634707].
- short_le5s, n59, Gamma025-Source: v+1.972104pp CI[-1.3570762782111292, 5.366001648870417]; t+3.904390 CI[-0.8059748581504959, 8.607877298731292].
- short_le5s, n59, Gamma05-Source: v+2.124738pp CI[-1.15469959112641, 5.448543783850435]; t+4.064864 CI[-0.5664566682276667, 8.714811986279681].
- short_le5s, n59, Gamma075-Source: v+1.835329pp CI[-1.467041293439374, 5.212675414840111]; t+3.780641 CI[-0.9188381410408359, 8.501949746215374].
- short_le5s, n59, Gamma1-Source: v+1.393810pp CI[-1.939070006325617, 4.8104903110810024]; t+3.179841 CI[-1.3284290096063505, 7.711664197059634].
- short_le5s, n59, Gamma0-Gamma05: v-0.355464pp CI[-1.0010526695074426, 0.31486972550982406]; t-0.512071 CI[-1.634689594830791, 0.4503486423548172].
- short_le5s, n59, Gamma025-Gamma05: v-0.152635pp CI[-0.6681344165400529, 0.3479785330573494]; t-0.160474 CI[-1.0904111432134296, 0.561845556430914].
- short_le5s, n59, Gamma075-Gamma05: v-0.289409pp CI[-0.7136223024556759, 0.1357934872561355]; t-0.284224 CI[-0.871525332702094, 0.31172535153895115].
- short_le5s, n59, Gamma1-Gamma05: v-0.730928pp CI[-1.4839549923966653, -0.04249990895318832]; t-0.885024 CI[-2.2318222052861456, 0.20584840164239385].
- long_gt5s, n133, Gamma0-Source: v+8.166317pp CI[5.331644882876846, 11.07247895170011]; t+12.175228 CI[9.077834080888438, 15.300907010366025].
- long_gt5s, n133, Gamma025-Source: v+8.638046pp CI[5.834632239944517, 11.495095256478443]; t+11.901092 CI[8.7751389682151, 15.098489831902288].
- long_gt5s, n133, Gamma05-Source: v+9.139312pp CI[6.224700975393956, 12.121539018028786]; t+11.817683 CI[8.754668120039954, 14.967892370695123].
- long_gt5s, n133, Gamma075-Source: v+9.577045pp CI[6.584792254534979, 12.621936960177042]; t+11.696783 CI[8.608510386564568, 14.911900159138826].
- long_gt5s, n133, Gamma1-Source: v+9.413877pp CI[6.301298753405211, 12.510389772706572]; t+11.733393 CI[8.57671538610605, 14.960708400904666].
- long_gt5s, n133, Gamma0-Gamma05: v-0.972996pp CI[-1.5133073894442874, -0.40448897645457793]; t+0.357544 CI[-0.18077326180097333, 0.9160019979548255].
- long_gt5s, n133, Gamma025-Gamma05: v-0.501267pp CI[-0.893579893234575, -0.11278685960496597]; t+0.083408 CI[-0.34948527044859146, 0.5039031630928987].
- long_gt5s, n133, Gamma075-Gamma05: v+0.437733pp CI[0.10834499489460651, 0.787288776463683]; t-0.120901 CI[-0.5615606949190579, 0.2460537887829234].
- long_gt5s, n133, Gamma1-Gamma05: v+0.274565pp CI[-0.32871600830923814, 0.88540646309782]; t-0.084290 CI[-0.7902600866432011, 0.6692773206607169].

Every gamma point reconstructs E, bidirectional geometry, actual native-grid E ROI features, prototype/posterior, own legal jointMAP interval and finalDP. Both spatial and temporal downstream effects change, so this is not an isolated appearance parameter. Current historical defaults remain HC0.5/Vid1; the complete five-point curve is descriptive sensitivity on exposed data and does not establish source-only calibration. No default selection, new fullGamma acquisition, old-owner restoration or main-method change.

All signed positive/negative rows, severe tails/rescues and short/long CIs are retained. Conditional replay elapsed clocks include file SHA, resource checks and actual original-grid pooling; not process CPU or complete deployment latency. New model/Source/expert/optimizer calls are zero. Original acquired Source/eight-expert contexts and old gamma preprocessing are inherited nonzero costs, not free observations. Shared original Vid producer/TTA costs must not be added twice.
