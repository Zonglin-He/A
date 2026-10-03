"""Source-backed human report and table parity receipt; no new fitting."""
import sys,json,csv,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tastvg_pr_nested_alpha/2026-10-03'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(OUT/'CONFIG.json');fits=read(OUT/'FIT_SUMMARY.json');cv=read(OUT/'INNER_CV.json');checks=0
def eq(a,b):
    global checks
    assert abs(float(a)-float(b))<3e-12,(a,b);checks+=1
rows=list(csv.DictReader((OUT/'REGRESSION.csv').open()))
for r in rows:
    z=read(OUT/r['dataset']/'SUMMARY.json')[r['panel']]['regression'][r['population']+'/'+r['regime']][r['role']]['metrics'][r['metric']]
    for k,v in [('mean',z['mean']),('ci_low',z['ci95'][0]),('ci_high',z['ci95'][1]),('undefined',z['bootstrap_undefined'])]:eq(r[k],v)
rs=list(csv.DictReader((OUT/'ALPHA_SELECTION.csv').open()))
for r in rs:
    m=next(m for m in fits if m['dataset']==r['dataset'] and m['held_source']==int(r['held_source']) and m['population']==r['population'])['heads'][r['head']]
    for k in ['alpha','training_MAE','training_MSE']:eq(r[k],m[k])
    eq(r['inner_MAE'],m['inner_source_MAE'][cfg['alpha_grid'].index(m['alpha'])])
rr=list(csv.DictReader((OUT/'INNER_SOURCE_MAE.csv').open()))
index={(c['dataset'],c['held_source'],c['population'],n,z['inner_source'],e['alpha']):e for c in cv for n,cc in c['heads'].items() for z in cc for e in z['scores']}
for r in rr:
    z=index[r['dataset'],int(r['outer_source']),r['population'],r['head'],int(r['inner_source']),float(r['alpha'])]
    for a,b in [('A_MAE','A'),('W_MAE','W'),('role_MAE','mean'),('clean_MAE','clean'),('corrupt_MAE','corrupt')]:eq(r[a],z[b])
assert len(rows)==768 and len(rs)==120 and len(rr)==11816
table=dict(status='pass',scalar_checks=checks,regression_rows=len(rows),alpha_rows=len(rs),inner_source_alpha_rows=len(rr),
    files={p:sha(OUT/p) for p in ['REGRESSION.csv','ALPHA_SELECTION.csv','INNER_SOURCE_MAE.csv']})
p=OUT/'TABLE_AUDIT.json'
if p.exists():assert read(p)==table
else:p.write_text(json.dumps(table,indent=2)+'\n')
root=read(OUT/'ROOT_AUDIT.json');public=read(OUT/'PUBLIC_AUDIT.json');assert root['status']==public['status']=='pass'
res=read(OUT/'RESOURCES.json');alpha=read(OUT/'ALPHA_DISTRIBUTION.json');roles=['P_A','R_A','P_W','R_W']
text=['# Nested Source-Regularization Audit：正则化有局部效应，未恢复普遍跨源质量读出\n',
'本轮按 bf45294 之后的用户授权完成。唯一科学变量是 ridge alpha 的选择方式：旧固定 alpha LOSO 对照，比较完全排除外层来源的 nested source-LOSO alpha 选择。不是新的无标签 TTA 方法，也没有接入 A 或 CURRENT_METHOD。\n',
'**主判断：旧 alpha 并非各读出的普遍最优值，HC recall 的收缩有可测改善；但这套 nested 选择没有把原来的跨源失败整体修复。** 16 个 corruption 角色/群体结果里，13 个 nested R² 点值仍为负；三个正点值的区间全部跨零，没有一个 R² 的95%区间完全在零以上。M-role 八个 R² 点值仍全负。不能据此宣布 final latent 没信息，也不能把所有失败唯一归于高维 variance。\n',
'## 数据、嵌套规则与控制\n',
'沿用已有 search 专家到达：VidSTG 16来源/96cells，HC-STVG-v2 14来源/96cells；每集80corrupt＋16clean。一query/source、两order、clean＋五类5% corruption、25% expert的原32-source开发设计不变，只用现有专家特征。确认不新拟合、不新评估，来源有历史曝光，holdout只针对质量读出的监督与normalizer；A持久轨迹可能无标签遇到被留出的source，不是fresh test。\n',
'TA-STVG官方同域checkpoint：Vid `'+cfg['checkpoint_state_sha256']['vidstg']+'`；HC `'+cfg['checkpoint_state_sha256']['hc2']+'`。原Paper48双offset/pixels，第6层Inside256→P、Endpoint512→R，A1792空间参数轨迹VidK1/HCK8、Old8⊂Expanded32、A8与固定L32 winner W全部不变。M-all训练all32，M-role训练[A,W]保留重复，两者共享各自的P/R heads。\n',
'外层16/14整源LOSO；内层只在其余15/13来源中逐源LOSO，所有该源的condition/order/candidate一起留出。alpha预锁 `[.001,.01,.1,1,10,100,1000]`，每个outer fold、population、P/R head分别选；选择目标是inner held-source **A/W双角色六条件等源MAE**，两个群体用同一部署角色目标。训练仍是原source-weighted MSE＋alpha系数范数惩罚，权重总和1、bias无罚、每个inner/outer训练集重估标准化；回归不clip，精确并列取最小alpha。主评测为corruption，所以all-condition selection与corrupt primary的差异是预锁设定，不是事后更换目标。\n',
'30outer folds、422ordered inner folds；11,816个inner candidate-head fits＋120个selected outer refits。七alpha复用同训练数据的eig分解，共1,808次，未改变特征空间或方程。没有对称fold复用。原nmax fixed-alpha预测逐值复用：Vid P/R=1/1，HC=1/.1。训练阶段file guard禁止读outer labelled pack，inner训练数组同时排除outer/inner source；全部选择/outer模型seal后，在独立GTfree进程读outer hidden预测并seal，最后才join已有GT用于评测。模型hash、输入/状态/candidate hash和全过程见RUNTIME_BINDING、FOLDS、INNER_CV、FIT_SUMMARY、GLOBAL_READOUT_SEAL、LABEL_JOIN。\n',
'## 主结果：同来源、同到达的 raw regression\n',
'MAE为0–1比例，差值是nested−fixed，负值较好；括号为10000次配对source-bootstrap95%区间。R²与Pearson均未裁剪。\n',
'| Dataset / population | Role | MAE fixed → nested | ΔMAE [95% CI] | R² fixed → nested | Pearson fixed → nested |',
'|---|---|---:|---:|---:|---:|']
for ds in ['vidstg','hc2']:
 s=read(OUT/ds/'SUMMARY.json')['corrupt']
 for pop in ['all','role']:
  for k in roles:
   a=s['regression'][pop+'/fixed'][k]['metrics'];b=s['regression'][pop+'/nested'][k]['metrics'];d=s['paired'][pop+'/nested minus '+pop+'/fixed']['regression'][k]['mae']
   text.append(f"| {ds} M-{pop} | {k} | {a['mae']['mean']:.4f} → {b['mae']['mean']:.4f} | {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {a['r2']['mean']:.4f} → {b['r2']['mean']:.4f} | {a['rho']['mean']:.4f} → {b['rho']['mean']:.4f} |")
text += ['\nHC R_W MAE有两项明确的点态配对改善：M-all .2676→.2283，差−.0393 [−.0721,−.0031]；M-role .2766→.2081，差−.0686 [−.1280,−.0093]。其余14个MAE差区间跨零，不能用“不显著”证明相等。HC四个recall ΔR²配对区间都高于零，但M-role R_A/R_W仍为−.1874/−.2055；M-all R_A的+.1167区间[−3.6837,+.3839]，并未建立稳定正R²。\n',
'特别是 **HC M-role recall 的MAE/MSE改善没有恢复正确的质量排序**：R_A Pearson −.6540 [−.8791,−.2972]，R_W −.7580 [−.9748,−.6235]。不能把绝对误差下降直接写成可靠quality signal，或者把它当成alpha已经解决mapping。\n',
'Vid M-all precision点值改善：P_A .3296→.2979，P_W .3854→.3566，但配对MAE/R²区间跨零；recall点值反而变坏。Vid M-role没有一致改善，P两项更坏、R两项MAE点值略降；其四R²仍为负。两dataset/两population不能事后拼选赢家。所有clean、order、condition以及未定义bootstrap draw完整保留在SUMMARY。\n',
'![Raw MAE](../results/tastvg_pr_nested_alpha/2026-10-03/figures/mae_fixed_vs_nested.png)\n',
'![Raw R2](../results/tastvg_pr_nested_alpha/2026-10-03/figures/r2_fixed_vs_nested.png)\n',
'## Alpha选择不是统一增强正则化\n',
'| Dataset | Population | P selected alpha: outer fold counts | R selected alpha: outer fold counts |',
'|---|---|---|---|']
for ds in ['vidstg','hc2']:
 for pop in ['all','role']:
  parts=['; '.join(f'{a}: {n}' for a,n in alpha[ds][pop][head].items() if n) for head in ['P','R']]
  text.append(f'| {ds} | M-{pop} | {parts[0]} | {parts[1]} |')
text += ['\nVid M-all P全部16折选.1，低于旧1；所以不能把旧拟合普遍描述为“正则化太弱”。HC M-all R在13/14折选10，HC M-role R在13/14折选1000，相对旧.1强很多；R_W MAE改善与这套选择相容，但不是variance为唯一原因的因果证明。M-role HC R大量命中预锁上界1000，只能说明在本grid里倾向更强收缩，不能排除范围外或另一selection objective；本轮不按外层结果扩grid。\n',
'![Alpha frequency](../results/tastvg_pr_nested_alpha/2026-10-03/figures/alpha_frequency.png)\n',
'完整每个inner source/alpha的A/W、clean/corrupt MAE在INNER_CV.json和INNER_SOURCE_MAE.csv，120外层head选择在FIT_SUMMARY.json/ALPHA_SELECTION.csv；选参只用内层来源，外层标签从未参与。\n',
'## Secondary：解析决策未普遍恢复\n',
'只在解析F中clip P/R，保持同一个eligible W，deltaF>0接受；无阈值搜索、无新candidate。以下Δv仅search corruption **专家子集**相对A，不是全流收益。\n',
'| Dataset / population | AUROC fixed → nested | BA fixed → nested | accepted helpful/harmful fixed → nested | severe accepts fixed → nested | expert Δv pp fixed → nested | nested−fixed Δv pp [95% CI] |',
'|---|---:|---:|---|---:|---:|---:|']
for ds in ['vidstg','hc2']:
 s=read(OUT/ds/'SUMMARY.json')['corrupt']
 for pop in ['all','role']:
  a=s['decision'][pop+'/fixed'];b=s['decision'][pop+'/nested'];d=s['paired'][pop+'/nested minus '+pop+'/fixed']['utility']['delta_v']
  text.append(f"| {ds} M-{pop} | {a['metrics']['auc']['mean']:.4f} → {b['metrics']['auc']['mean']:.4f} | {a['metrics']['balanced_accuracy']['mean']:.4f} → {b['metrics']['balanced_accuracy']['mean']:.4f} | {a['counts']['accepted_helpful']}/{a['counts']['accepted_harmful']} → {b['counts']['accepted_helpful']}/{b['counts']['accepted_harmful']} | {a['counts']['accepted_severe_v_harm']} → {b['counts']['accepted_severe_v_harm']} | {a['utility']['delta_v']['mean']*100:+.4f} → {b['utility']['delta_v']['mean']*100:+.4f} | {d['mean']*100:+.4f} [{d['ci95'][0]*100:+.4f}, {d['ci95'][1]*100:+.4f}] |")
text += ['\nVid M-role BA下降 .2031 [−.3796,−.0384]，配对expert vIoU下降 .3882pp [−.8991,−.0265]；有害accept24→30、严重8→9。这是本固定解析执行的负结果，不能因为回归某项MAE较低而忽略。HC M-role有益accept22→26、有害13不变，但Δv+.3136pp [−.0167,+.8791]仍不确定。所有AUROC变化区间跨零。\n',
'## Work / failure cases，均为事后解释\n',
'公开CASES.json同时保存误差改善最大/恶化最大，以及决策变化的正负样本；下面不是线上筛选规则。cell key末段是arrival，source_id以ROWS字段为准。\n',
'| Dataset / population / cell key | 四角色平均绝对误差 fixed → nested | 解析决定 fixed → nested | 固定W真实Δv pp |',
'|---|---:|---|---:|']
keys=[('vidstg','all','vidstg/search/exposure_5/order2/12'),('vidstg','all','vidstg/search/occlusion_5/order2/20'),
 ('vidstg','role','vidstg/search/motion_blur_5/order2/12'),('vidstg','role','vidstg/search/exposure_5/order2/24'),
 ('vidstg','role','vidstg/search/motion_blur_5/order2/24'),('hc2','all','hc2/search/exposure_5/order2/20'),
 ('hc2','all','hc2/search/motion_blur_5/order2/16'),('hc2','role','hc2/search/occlusion_5/order1/12')]
rowmap={r['cell_key']:r for r in read(OUT/'ROWS.json')}
for ds,pop,key in keys:
 r=rowmap[key];err=lambda reg:sum(abs(r['truth'][k]-r['readouts'][pop+'/'+reg][k]) for k in roles)/4
 take=lambda reg:'accept' if r['analytic'][pop+'/'+reg]['accepted'] else 'reject'
 text.append(f"| {ds} M-{pop} source{r['source_id']} `{key}` | {err('fixed'):.4f} → {err('nested'):.4f} | {take('fixed')} → {take('nested')} | {r['delta_v']*100:+.4f} |")
text += ['\nVid M-all exposure误差 .3818→.2171，仍接受真实较差W：MAE改善不自动保证delta判别。Vid M-all occlusion拒掉−18.53pp的大坏例，保留该正向保护。Vid M-role source3 exposure原正确reject变为accept坏W（−7.28pp）；同source blur原accept有益W（+9.82pp）变reject，说明不能把收益只归于更保守。HC M-all exposure误差 .5259→.3997仍接受−10.55pp坏W；motion_blur原较准的P被压低，误差 .1363→.2976但仍接受有益W。HC M-role occlusion由误拒变接受+11.87pp；这些正负例均未进入alpha选择。\n',
'## 核验、成本与判断边界\n',
f"10项CPU测试通过；根审计独立用SciPy SPD求解复核全部11,816 inner候选和120 outer模型，与生产NumPy eigen实现交叉检查，总{root['total_scalar_checks']:,}标量/结构检查；公开审计{public['total_scalar_checks']:,}项，CSV三表{checks:,}标量核对。物理P/R/tIoU、源权重、normalizer、outer/inner排除、选择、固定control逐值相同、candidate/state/pixel hash和10000source配对区间均核验。三PNG已目检，无重叠/裁切；PDF同步生成。\n",
f"科学计算CPU wall：拟合{res['fit_CPU_wall_seconds']:.3f}s，GTfree读出{res['readout_CPU_wall_seconds']:.3f}s，诊断{res['diagnosis_CPU_wall_seconds']:.3f}s；独立根审计{root['CPU_wall_seconds']:.3f}s、公开审计{public['CPU_wall_seconds']:.3f}s另列。不含代码开发/报告/绘图/远端同步，零GPU、专家、backbone、candidate、replay、backward、新在线流或production调用。\n",
'Bootstrap不重fit，条件于固定OOF预测；outer模型训练集重叠，source仅16/14，点态区间未多重校正。不能用三项正R²点值或两项MAE区间宣称全面解决，不能用其余区间跨零宣布参数无效。固定上界alpha命中也不证明所有regularization探索已穷尽。历史曝光与多轮开发不等于未见test。\n',
'**决策：保留 A 与 CURRENT；nested-alpha读出不接入正式方法。** 本轮把“旧alpha选择不合适”从未测试因素变成了已测的局部因素：HC recall存在改善，Vid不一致，可靠A/W质量mapping和安全decision未普遍恢复。停止把“调大alpha即可救回”作为已成立故事，继续保留regularization、独立source数与representation-conditioned mapping之间的不确定性。没有启动PCA/低秩/layerwise/MLP/新gate/新专家/新队列；后续新实验需要明确机制与匹配控制。\n',
'## 可复核文件\n',
'[协议](../protocols/tastvg_pr_nested_alpha_v1.md)、[执行说明](tastvg_pr_nested_alpha_v1/EXECUTION.md)、[全部匿名结果](../results/tastvg_pr_nested_alpha/2026-10-03)、[根审计](../results/tastvg_pr_nested_alpha/2026-10-03/ROOT_AUDIT.json)、[公开审计](../results/tastvg_pr_nested_alpha/2026-10-03/PUBLIC_AUDIT.json)、[完整内层曲线](../results/tastvg_pr_nested_alpha/2026-10-03/INNER_CV.json)。私有hidden/模型/normalizer/GT时间span/媒体均不公开。\n']
p=ROOT/'docs/TA_PR_NESTED_REGULARIZATION_REVIEW.md';assert not p.exists();p.write_text('\n'.join(text))
print('Report generated; table parity',checks,'checks.')
