# 实际时间观察：HC128冻结空间能力筛查

同一TA-STVG Source权重，全部RC128已曝光开发父，每父一query。原空间Live41/42/43先父内均值，E辅助；任何输出不改原空间坐标。比较同覆盖较粗33实际帧和围绕原MBR扩2x上下文的最多65实际帧重读；不是缓存特征裁剪，不增加DINO/Refer或训练。

| 时间读出 | vIoU% | tIoU% | v增量pp [95%CI] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| CoarseDirect | 27.907784 | 52.040080 | -0.020690 [-2.087398,+2.046143] |
| WindowDirect | 26.377179 | 48.400964 | -1.551295 [-2.551944,-0.578443] |
| CoarseMix | 28.535603 | 53.004205 | +0.607129 [-0.725881,+1.907555] |
| WindowMix | 27.168738 | 49.896431 | -0.759736 [-1.362440,-0.180168] |
| CoarseNoRGBDirect | 27.009975 | 49.825247 | -0.918499 [-2.003984,+0.023745] |
| CoarseNoRGBMix | 27.391713 | 50.599432 | -0.536761 [-1.186890,+0.016460] |
| WindowNoRGBDirect | 27.665763 | 50.828229 | -0.262711 [-0.527348,-0.011582] |
| WindowNoRGBMix | 27.774653 | 51.085913 | -0.153821 [-0.337999,+0.023190] |

Direct用新观察后验；Mix固定.5原分布+.5新观察后验。NoRGB对照仅将旧各offset端点标准化logits按物理时间插值到新采样点，重建同支持后验，不重编码RGB；用于判断是否仅限制候选/重采样logits，并非真实新视觉证据或选型候选。该对照在前两预测/后续运行中、任何新GT之前增补，原五arm预测不改。局部后验不是事件存在概率，全局保留不承诺无损。所有组最终决策严格原65物理端点2080区间；仅hypothesis物理支持union用于精确expectedIoU，未增加可输出决策。Native及grid-only嵌入精确恢复旧MBR，空间s按构造完全一致。

## 开发选择与边界

{
  "selected_development_candidate": "CoarseMix",
  "selection": "highest fullv parent seed mean across4 prespecified views/readouts; ties preferMix then lexical; development selection not independent proof",
  "selected_delta": {
    "mean": 0.6071287056910393,
    "CI95": [
      -0.7258810292805921,
      1.9075553955395759
    ],
    "parents": 128
  },
  "decision": "no_candidate_has_supported_meaningful_development_increment",
  "confirmation_not_started": true,
  "not_OPD": true
}

四个预登记候选的pointwise区间不修正开发选择偏差；选中后须父不重叠冻结确认。没有按视频/错误类别挑赢家或删样本，不把观察/读出增益称OPD。

## 全部尾部

### CoarseDirect

```json
{
  "positive": 64,
  "negative": 60,
  "flat": 4,
  "below_minus3": 45,
  "below_minus5": 33,
  "positive_contribution_pp": 4.133815215833794,
  "negative_contribution_pp": -4.154505297473127,
  "worst": [
    {
      "alias": "RC042",
      "delta_pp": -39.011519528270306
    },
    {
      "alias": "RC082",
      "delta_pp": -34.966101101190866
    },
    {
      "alias": "RC111",
      "delta_pp": -32.38323402008958
    },
    {
      "alias": "RC056",
      "delta_pp": -21.379823690396126
    },
    {
      "alias": "RC118",
      "delta_pp": -21.308814966362398
    },
    {
      "alias": "RC036",
      "delta_pp": -20.57680121949393
    },
    {
      "alias": "RC080",
      "delta_pp": -19.627980251366374
    },
    {
      "alias": "RC096",
      "delta_pp": -18.76568540977065
    }
  ]
}
```

### WindowDirect

```json
{
  "positive": 29,
  "negative": 52,
  "flat": 47,
  "below_minus3": 29,
  "below_minus5": 19,
  "positive_contribution_pp": 0.7494586081323003,
  "negative_contribution_pp": -2.300753415114956,
  "worst": [
    {
      "alias": "RC056",
      "delta_pp": -29.274434352389495
    },
    {
      "alias": "RC099",
      "delta_pp": -21.189267764341167
    },
    {
      "alias": "RC015",
      "delta_pp": -16.64286032831593
    },
    {
      "alias": "RC096",
      "delta_pp": -16.514203372411057
    },
    {
      "alias": "RC065",
      "delta_pp": -16.339338731014802
    },
    {
      "alias": "RC016",
      "delta_pp": -15.824935421927455
    },
    {
      "alias": "RC009",
      "delta_pp": -15.47281607251016
    },
    {
      "alias": "RC036",
      "delta_pp": -12.159987459236959
    }
  ]
}
```

### CoarseMix

```json
{
  "positive": 69,
  "negative": 50,
  "flat": 9,
  "below_minus3": 24,
  "below_minus5": 18,
  "positive_contribution_pp": 2.5792308915287614,
  "negative_contribution_pp": -1.9721021858377221,
  "worst": [
    {
      "alias": "RC042",
      "delta_pp": -35.47776558224107
    },
    {
      "alias": "RC111",
      "delta_pp": -26.395121481664425
    },
    {
      "alias": "RC014",
      "delta_pp": -15.2411731932176
    },
    {
      "alias": "RC082",
      "delta_pp": -11.735785059259918
    },
    {
      "alias": "RC026",
      "delta_pp": -11.583509230058652
    },
    {
      "alias": "RC118",
      "delta_pp": -10.96082772690693
    },
    {
      "alias": "RC044",
      "delta_pp": -9.339763008405633
    },
    {
      "alias": "RC100",
      "delta_pp": -9.170021302129683
    }
  ]
}
```

### WindowMix

```json
{
  "positive": 29,
  "negative": 46,
  "flat": 53,
  "below_minus3": 17,
  "below_minus5": 11,
  "positive_contribution_pp": 0.5261159848030951,
  "negative_contribution_pp": -1.2858521666048845,
  "worst": [
    {
      "alias": "RC056",
      "delta_pp": -16.800887186731135
    },
    {
      "alias": "RC099",
      "delta_pp": -12.033238048130729
    },
    {
      "alias": "RC065",
      "delta_pp": -12.017002312707387
    },
    {
      "alias": "RC016",
      "delta_pp": -10.82549180932071
    },
    {
      "alias": "RC015",
      "delta_pp": -10.106640576852644
    },
    {
      "alias": "RC096",
      "delta_pp": -8.54657865836978
    },
    {
      "alias": "RC036",
      "delta_pp": -7.674554418028073
    },
    {
      "alias": "RC128",
      "delta_pp": -6.276644004818399
    }
  ]
}
```

## 工程、计费与核验

原900s自定预算在首两无GT计时max外推995.8s时按规则停止，2完整预测与成本保留；仅修订有限总预算1200s/单进程900s续跑，名单/观察规则/公式/参数/Source调用上限不变，不读取GT后救参数。准备阶段合成Direct测试发现union漏纳原决策端点，在任何模型/GT前修正为协议要求，旧码/plan保留preflight001。
```json
{
  "GPU_seconds": 945.9328693819698,
  "new_Source_view_queries": 224,
  "Source_view_offset_forwards": 448,
  "exact_duplicate_views_reused": 32,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "network_weight_updates": 0,
  "spatial_updates": 0,
  "case_costs": [
    {
      "alias": "RC001",
      "seconds": 7.4681112370453775,
      "new_Source": 2
    },
    {
      "alias": "RC002",
      "seconds": 3.6412161050830036,
      "new_Source": 1
    },
    {
      "alias": "RC003",
      "seconds": 7.137346409028396,
      "new_Source": 2
    },
    {
      "alias": "RC004",
      "seconds": 6.904479868942872,
      "new_Source": 2
    },
    {
      "alias": "RC005",
      "seconds": 7.208904011873528,
      "new_Source": 2
    },
    {
      "alias": "RC006",
      "seconds": 9.478200206998736,
      "new_Source": 2
    },
    {
      "alias": "RC007",
      "seconds": 7.107277140952647,
      "new_Source": 2
    },
    {
      "alias": "RC008",
      "seconds": 6.908228907966986,
      "new_Source": 2
    },
    {
      "alias": "RC009",
      "seconds": 9.999026070116088,
      "new_Source": 2
    },
    {
      "alias": "RC010",
      "seconds": 7.372750377980992,
      "new_Source": 2
    },
    {
      "alias": "RC011",
      "seconds": 7.623840554151684,
      "new_Source": 2
    },
    {
      "alias": "RC012",
      "seconds": 7.458361790981144,
      "new_Source": 2
    },
    {
      "alias": "RC013",
      "seconds": 7.273666444933042,
      "new_Source": 2
    },
    {
      "alias": "RC014",
      "seconds": 9.72337721590884,
      "new_Source": 2
    },
    {
      "alias": "RC015",
      "seconds": 7.369228744180873,
      "new_Source": 2
    },
    {
      "alias": "RC016",
      "seconds": 9.926402406999841,
      "new_Source": 2
    },
    {
      "alias": "RC017",
      "seconds": 6.877182413823903,
      "new_Source": 2
    },
    {
      "alias": "RC018",
      "seconds": 7.823169466108084,
      "new_Source": 2
    },
    {
      "alias": "RC019",
      "seconds": 3.621589185902849,
      "new_Source": 1
    },
    {
      "alias": "RC020",
      "seconds": 7.080295090097934,
      "new_Source": 2
    },
    {
      "alias": "RC021",
      "seconds": 4.0842452850192785,
      "new_Source": 1
    },
    {
      "alias": "RC022",
      "seconds": 7.649586479878053,
      "new_Source": 2
    },
    {
      "alias": "RC023",
      "seconds": 6.903526509879157,
      "new_Source": 2
    },
    {
      "alias": "RC024",
      "seconds": 7.368154268944636,
      "new_Source": 2
    },
    {
      "alias": "RC025",
      "seconds": 4.711618663975969,
      "new_Source": 1
    },
    {
      "alias": "RC026",
      "seconds": 10.301643359009176,
      "new_Source": 2
    },
    {
      "alias": "RC027",
      "seconds": 5.259351014858112,
      "new_Source": 1
    },
    {
      "alias": "RC028",
      "seconds": 3.666210788069293,
      "new_Source": 1
    },
    {
      "alias": "RC029",
      "seconds": 6.852714397013187,
      "new_Source": 2
    },
    {
      "alias": "RC030",
      "seconds": 7.228605075040832,
      "new_Source": 2
    },
    {
      "alias": "RC031",
      "seconds": 6.953446680912748,
      "new_Source": 2
    },
    {
      "alias": "RC032",
      "seconds": 4.035329484147951,
      "new_Source": 1
    },
    {
      "alias": "RC033",
      "seconds": 3.542496561072767,
      "new_Source": 1
    },
    {
      "alias": "RC034",
      "seconds": 7.368327066069469,
      "new_Source": 2
    },
    {
      "alias": "RC035",
      "seconds": 7.788555952953175,
      "new_Source": 2
    },
    {
      "alias": "RC036",
      "seconds": 15.964175967965275,
      "new_Source": 2
    },
    {
      "alias": "RC037",
      "seconds": 7.225812803953886,
      "new_Source": 2
    },
    {
      "alias": "RC038",
      "seconds": 7.149954105960205,
      "new_Source": 2
    },
    {
      "alias": "RC039",
      "seconds": 6.982145334826782,
      "new_Source": 2
    },
    {
      "alias": "RC040",
      "seconds": 7.9839354059658945,
      "new_Source": 2
    },
    {
      "alias": "RC041",
      "seconds": 5.192184029845521,
      "new_Source": 1
    },
    {
      "alias": "RC042",
      "seconds": 6.388636176008731,
      "new_Source": 2
    },
    {
      "alias": "RC043",
      "seconds": 7.005384815856814,
      "new_Source": 2
    },
    {
      "alias": "RC044",
      "seconds": 7.078553257975727,
      "new_Source": 2
    },
    {
      "alias": "RC045",
      "seconds": 14.876198571175337,
      "new_Source": 2
    },
    {
      "alias": "RC046",
      "seconds": 6.658646667841822,
      "new_Source": 2
    },
    {
      "alias": "RC047",
      "seconds": 3.2924626010935754,
      "new_Source": 1
    },
    {
      "alias": "RC048",
      "seconds": 7.1599746169522405,
      "new_Source": 2
    },
    {
      "alias": "RC049",
      "seconds": 6.968664223095402,
      "new_Source": 2
    },
    {
      "alias": "RC050",
      "seconds": 3.6568013180512935,
      "new_Source": 1
    },
    {
      "alias": "RC051",
      "seconds": 6.633760657161474,
      "new_Source": 2
    },
    {
      "alias": "RC052",
      "seconds": 3.948400278110057,
      "new_Source": 1
    },
    {
      "alias": "RC053",
      "seconds": 5.491868424927816,
      "new_Source": 1
    },
    {
      "alias": "RC054",
      "seconds": 3.8264880019705743,
      "new_Source": 1
    },
    {
      "alias": "RC055",
      "seconds": 6.3992586641106755,
      "new_Source": 2
    },
    {
      "alias": "RC056",
      "seconds": 8.887147854780778,
      "new_Source": 2
    },
    {
      "alias": "RC057",
      "seconds": 6.306564165977761,
      "new_Source": 2
    },
    {
      "alias": "RC058",
      "seconds": 7.410372639074922,
      "new_Source": 2
    },
    {
      "alias": "RC059",
      "seconds": 3.529281812021509,
      "new_Source": 1
    },
    {
      "alias": "RC060",
      "seconds": 7.117010862100869,
      "new_Source": 2
    },
    {
      "alias": "RC061",
      "seconds": 6.917220973875374,
      "new_Source": 2
    },
    {
      "alias": "RC062",
      "seconds": 3.7629047790542245,
      "new_Source": 1
    },
    {
      "alias": "RC063",
      "seconds": 17.452132034115493,
      "new_Source": 2
    },
    {
      "alias": "RC064",
      "seconds": 6.984673036960885,
      "new_Source": 2
    },
    {
      "alias": "RC065",
      "seconds": 7.082253336906433,
      "new_Source": 2
    },
    {
      "alias": "RC066",
      "seconds": 6.448013362940401,
      "new_Source": 2
    },
    {
      "alias": "RC067",
      "seconds": 6.543259390862659,
      "new_Source": 2
    },
    {
      "alias": "RC068",
      "seconds": 5.110995364142582,
      "new_Source": 1
    },
    {
      "alias": "RC069",
      "seconds": 6.248213740997016,
      "new_Source": 2
    },
    {
      "alias": "RC070",
      "seconds": 3.3221779197920114,
      "new_Source": 1
    },
    {
      "alias": "RC071",
      "seconds": 3.7133281661663204,
      "new_Source": 1
    },
    {
      "alias": "RC072",
      "seconds": 6.788366917986423,
      "new_Source": 2
    },
    {
      "alias": "RC073",
      "seconds": 8.009992155013606,
      "new_Source": 2
    },
    {
      "alias": "RC074",
      "seconds": 6.612977457931265,
      "new_Source": 2
    },
    {
      "alias": "RC075",
      "seconds": 6.553136612987146,
      "new_Source": 2
    },
    {
      "alias": "RC076",
      "seconds": 15.283823620062321,
      "new_Source": 2
    },
    {
      "alias": "RC077",
      "seconds": 7.07803497207351,
      "new_Source": 2
    },
    {
      "alias": "RC078",
      "seconds": 3.482049335958436,
      "new_Source": 1
    },
    {
      "alias": "RC079",
      "seconds": 7.027867660857737,
      "new_Source": 2
    },
    {
      "alias": "RC080",
      "seconds": 6.484324879013002,
      "new_Source": 2
    },
    {
      "alias": "RC081",
      "seconds": 7.4216374000534415,
      "new_Source": 2
    },
    {
      "alias": "RC082",
      "seconds": 6.7119513330981135,
      "new_Source": 2
    },
    {
      "alias": "RC083",
      "seconds": 7.245484855957329,
      "new_Source": 2
    },
    {
      "alias": "RC084",
      "seconds": 4.1103779000695795,
      "new_Source": 1
    },
    {
      "alias": "RC085",
      "seconds": 6.476376017788425,
      "new_Source": 2
    },
    {
      "alias": "RC086",
      "seconds": 14.167186944046989,
      "new_Source": 2
    },
    {
      "alias": "RC087",
      "seconds": 8.864559686044231,
      "new_Source": 2
    },
    {
      "alias": "RC088",
      "seconds": 9.019199301954359,
      "new_Source": 2
    },
    {
      "alias": "RC089",
      "seconds": 6.687129453988746,
      "new_Source": 1
    },
    {
      "alias": "RC090",
      "seconds": 3.51410850812681,
      "new_Source": 1
    },
    {
      "alias": "RC091",
      "seconds": 5.671179398195818,
      "new_Source": 1
    },
    {
      "alias": "RC092",
      "seconds": 6.870186656014994,
      "new_Source": 2
    },
    {
      "alias": "RC093",
      "seconds": 8.947777095949277,
      "new_Source": 2
    },
    {
      "alias": "RC094",
      "seconds": 7.180922624887899,
      "new_Source": 2
    },
    {
      "alias": "RC095",
      "seconds": 6.402559546986595,
      "new_Source": 2
    },
    {
      "alias": "RC096",
      "seconds": 6.884335421025753,
      "new_Source": 2
    },
    {
      "alias": "RC097",
      "seconds": 3.7480742149055004,
      "new_Source": 1
    },
    {
      "alias": "RC098",
      "seconds": 5.163155338959768,
      "new_Source": 1
    },
    {
      "alias": "RC099",
      "seconds": 7.03319504391402,
      "new_Source": 2
    },
    {
      "alias": "RC100",
      "seconds": 6.861289876978844,
      "new_Source": 2
    },
    {
      "alias": "RC101",
      "seconds": 6.636607109103352,
      "new_Source": 2
    },
    {
      "alias": "RC102",
      "seconds": 7.29083103989251,
      "new_Source": 2
    },
    {
      "alias": "RC103",
      "seconds": 6.952134194783866,
      "new_Source": 2
    },
    {
      "alias": "RC104",
      "seconds": 3.8859347021207213,
      "new_Source": 1
    },
    {
      "alias": "RC105",
      "seconds": 9.56384636904113,
      "new_Source": 2
    },
    {
      "alias": "RC106",
      "seconds": 6.833171488950029,
      "new_Source": 2
    },
    {
      "alias": "RC107",
      "seconds": 4.486111214151606,
      "new_Source": 1
    },
    {
      "alias": "RC108",
      "seconds": 6.411853186087683,
      "new_Source": 2
    },
    {
      "alias": "RC109",
      "seconds": 6.966871963115409,
      "new_Source": 2
    },
    {
      "alias": "RC110",
      "seconds": 3.522769137052819,
      "new_Source": 1
    },
    {
      "alias": "RC111",
      "seconds": 6.441914050141349,
      "new_Source": 2
    },
    {
      "alias": "RC112",
      "seconds": 3.5151539959479123,
      "new_Source": 1
    },
    {
      "alias": "RC113",
      "seconds": 7.035300927003846,
      "new_Source": 2
    },
    {
      "alias": "RC114",
      "seconds": 7.2135445040185004,
      "new_Source": 2
    },
    {
      "alias": "RC115",
      "seconds": 6.983976111980155,
      "new_Source": 2
    },
    {
      "alias": "RC116",
      "seconds": 6.5767351891845465,
      "new_Source": 2
    },
    {
      "alias": "RC117",
      "seconds": 15.496466106968,
      "new_Source": 2
    },
    {
      "alias": "RC118",
      "seconds": 6.625597785925493,
      "new_Source": 2
    },
    {
      "alias": "RC119",
      "seconds": 9.745441135950387,
      "new_Source": 2
    },
    {
      "alias": "RC120",
      "seconds": 9.046826626872644,
      "new_Source": 2
    },
    {
      "alias": "RC121",
      "seconds": 6.44902705703862,
      "new_Source": 2
    },
    {
      "alias": "RC122",
      "seconds": 3.649539757054299,
      "new_Source": 1
    },
    {
      "alias": "RC123",
      "seconds": 3.993158257100731,
      "new_Source": 1
    },
    {
      "alias": "RC124",
      "seconds": 6.764274618122727,
      "new_Source": 2
    },
    {
      "alias": "RC125",
      "seconds": 6.985750732012093,
      "new_Source": 2
    },
    {
      "alias": "RC126",
      "seconds": 7.09734448720701,
      "new_Source": 2
    },
    {
      "alias": "RC127",
      "seconds": 9.787699677981436,
      "new_Source": 2
    },
    {
      "alias": "RC128",
      "seconds": 9.327545036096126,
      "new_Source": 2
    }
  ],
  "attempts": [
    {
      "GPU_seconds": 24.136500569991767,
      "new_Source_view_queries": 3,
      "Source_view_offset_forwards": 6,
      "exact_duplicate_views_reused": 1,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "network_weight_updates": 0,
      "spatial_updates": 0
    },
    {
      "GPU_seconds": 899.0442217600066,
      "new_Source_view_queries": 219,
      "Source_view_offset_forwards": 438,
      "exact_duplicate_views_reused": 31,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "network_weight_updates": 0,
      "spatial_updates": 0
    },
    {
      "GPU_seconds": 22.752147051971406,
      "new_Source_view_queries": 2,
      "Source_view_offset_forwards": 4,
      "exact_duplicate_views_reused": 0,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "network_weight_updates": 0,
      "spatial_updates": 0
    }
  ]
}
```

独立核验：
```json
{
  "status": "pass",
  "checks": {
    "Source_posteriors": 256,
    "physical_mass_mixtures": 640,
    "common_decision_grid": 640,
    "direct_utilities": 10880,
    "spatial_arrays_exact": 512,
    "grid_only_exact": 128
  },
  "max_independent_error": 8.881784197001252e-16,
  "exact_duplicate_observations": 32,
  "mean_new_context_RGB_frames": 42.296875,
  "original_spatial_and_native_inputs_unchanged": true,
  "all128_sealed_before_GT": true,
  "all128_all5arms_retained": true,
  "new_experts": 0,
  "new_network_training": 0,
  "all128_no_RGB_controls_sealed_before_GT": true
}
```

历史密采样HC负结果与对象memory负结果保留。本轮判断仅限这两类真实时间观察和固定读出，不泛化所有视频条件OPD。当前研究/生产登记及空间配置不改；全部逐父×3seed×5时间/4固定空间signed/unknown在RESULTS_PUBLIC.json，RGB/GT坐标/文本/身份/权重/raw caches不公开。
