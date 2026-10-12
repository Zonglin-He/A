"""Complete anonymous controlled-state report and two scientific PDF artifacts."""
import os,sys,csv,time,math
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_LN_state_intervention_common_v1 import *
from scripts.score_stvg_opd_LN_state_intervention_v1 import rows_read,estimate
def fmt(e,pp=True):
 if e['mean'] is None:return 'undefined'
 scale=100 if pp else 1
 return f"{e['mean']*scale:+.3f} ["+', '.join(f'{v*scale:+.3f}' for v in e['ci95'])+']'
def run():
 verify();root=read(BASE/'ROOT_READBACK.json');assert root['status']=='pass';rr=rows_read(OUT/'ROWS.jsonl.gz');s=read(OUT/'SUMMARY.json');decision=read(BASE/'MECHANISM_DECISION.json')
 with (OUT/'ALL_PARENT_THREE_ARM_READOUTS.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 report=['# Controlled LN State Intervention','',
  'One fixed historically exposed diagnostic panel: the explicitly selected P9/P0 previous-P6 32 parents in HC and 32 in VidSTG. No score-based case selection; zero query overlap with the separate TENT/SAR calibration 32. This is a controlled mechanism study, not a held-out performance test.','',
  'Three arms per query: Source LN, inherited LN from original order1, inherited LN from original order2. Only six LN tensors (1536 coordinates) change at entry. Query Residual starts at exact zero and Adam is fresh. The original joint 1792-coordinate OPD update, final round, clean pixels, source checkpoint, decoder inputs, Omega0 teacher evidence, admission, positions and antithetic noise remain fixed. Actions and gradients can depend on state. There is no new cross-query writeback.','',
  'All24 actual complete qualification episodes used two genuinely fresh frozen captures per fixed query and exact complete repeats. All192 formal episodes/4800 rounds are new full fits. The128 inherited formal fits exactly reproduce the corresponding old original Full episodes;12 formal fits exactly match their qualification fits. The21 empty formal episodes (7 Vid queries x3) have no sampler or optimizer and remain in every table. All predictions were sealed globally before raw GT scoring. No new DINO call, teacher, gate, tuning or production/main experiment.','',
  'Primary U comprises genuine annotated decoder frames within the GT event, excluding every actually visited Omega0 observation (including rejected teachers). Secondary common-U also excludes Omega1 visits for P0 comparability. No dense interpolation is used to manufacture genuine U. Before/After are computed separately for each initial state, with an identical mask across arms. Empty U stays unknown. Full dense s/v/t and the observed region are diagnostics. HC preserves the original coordinate lower-bound clipping.','',
  'All estimates use32 parent units per domain,10000 fixed paired parent bootstrap draws, seed20261013 and percentile95% intervals. Three prespecified contrasts and negative/neutral/unknown results are fully reported. Intervals and fixed1/5/20pp tiers are descriptive, with no multiple-comparison adjustment or held-out claim.','',
  '|Target|Initial state|U Before (%)|U After (%)|Update utility (pp),95% CI|Positive/negative/neutral/unknown|Harm <-5pp/<-20pp|',
  '|---|---|---|---|---|---|---|']
 derived={}
 for ds in ['hc2','vidstg']:
  rows=[r for r in rr if r['dataset']==ds];derived[ds]={}
  for arm in ARMS:
   a=s['datasets'][ds]['own_U']['arms'][arm];before=estimate([r[arm+'_own_U_before'] for r in rows]);after=estimate([r[arm+'_own_U_after'] for r in rows]);derived[ds][arm]=dict(before=before,after=after)
   report.append(f'|{ds}|{arm}|{fmt(before)}|{fmt(after)}|{fmt(a["estimate"])}|{a["positive_parents"]}/{a["negative_parents"]}/{a["neutral_parents"]}/{a["unknown_parents"]}|{a["harm5pp"]}/{a["harm20pp"]}|')
 report += ['','|Target|Contrast (first -> second)|Mean utility difference (pp),95% CI|Mean absolute difference (pp),95% CI|Flips|Both >=1pp/>=5pp|Difference >=5pp/>=20pp|Unknown|','|---|---|---|---|---|---|---|---|']
 for ds in ['hc2','vidstg']:
  for key,v in s['datasets'][ds]['own_U']['contrasts'].items():
   report.append(f'|{ds}|{key}|{fmt(v["signed_second_minus_first"])}|{fmt(v["absolute_difference"])}|{v["sign_flips"]}/32|{v["flip_both_at_least1pp"]}/{v["flip_both_at_least5pp"]}|{v["differences_at_least5pp"]}/{v["differences_at_least20pp"]}|{v["unknown_pairs"]}|')
 report += ['','The initial-state effect is evaluated on each arm\'s own Before-to-After utility, not on final accuracy alone. A signed mean near zero can coexist with material causal changes in opposite directions. Under the predeclared rule, a positive lower interval bound for absolute utility difference plus material sign reversals or >=5pp differences supports receiving-state dependence on this panel.','']
 for ds,v in decision['datasets'].items():report.append(f'- {ds}: fixed-panel material LN state dependence supported = {v["fixed_panel_material_LN_state_dependence_supported"]}; qualifying prespecified contrasts: '+', '.join(v['supported_contrasts'])+'.')
 report += ['','This is a causal intervention on the saved entering LN vectors while the remaining source model/input/evidence/update/noise are fixed. The exact repeat and original-reproduction controls eliminate a capture or replay change as the source of these differences. It does not prove generalization to unseen parents, identify which LN subcomponents cause the effect, establish a universal harm of inheritance, or provide a reliable unlabelled decision signal. No GT-selected state or step is used online.','',
  'Inherited states already have different Before accuracy. For example, Vid order1 starts at40.716% versus35.980% for Source, and finishes at48.755% versus47.784%, while its update utility is8.039pp versus11.804pp. Smaller update utility therefore does not automatically mean worse final prediction. This total LN-initialization effect does not separately identify initial-position/headroom effects versus changed gradients or decoder sensitivity. Same noise also does not mean same candidate boxes: the original state-dependent mean changes sampled actions and their teacher rewards.','',
  'The result freezes this experiment. No automatic gate, predictor, prototype, new loss, wider state search, paper drafting or CURRENT_METHOD promotion follows. P7-A remains frozen. Cancelled TENT/SAR formal evaluation remains cancelled with its original accepted prefix/optimizer/process identity in T standby.','',
  f'Independent root: {root["counts"]["fits"]} complete fits, {root["counts"]["rounds"]} rounds, {root["counts"]["independent_Adam_scalar_checks"]} independent Adam scalar checks, {root["counts"]["sample_coordinates"]} exact replayed sample coordinates, {root["independent_scalar_checks"]} independent outcome/statistic checks. Full model forward was actually run; the independent CPU root does not independently replay the full decoder Jacobian. Actual native head VJP was checked during qualification.','',
  'Complete anonymous64-parent x3-arm Before/After/utility/real-mask counts/empty-episode/full dense diagnostics are in ROWS.jsonl.gz and ALL_PARENT_THREE_ARM_READOUTS.csv. All domain/arm/contrast/transition/tail/CI statistics are in SUMMARY.json. Private model states, raw fits, samples, teacher/GT geometry, media and text stay local.']
 (OUT/'REPORT.md').write_text('\n'.join(report)+'\n');write(OUT/'BEFORE_AFTER_PARENT_SUMMARY.json',derived)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42})
 figdir=OUT/'figures';figdir.mkdir(exist_ok=False);pdfs=[]
 p1=figdir/'CONTROLLED_LN_UPDATE_UTILITY.pdf'
 with PdfPages(p1) as pdf:
  fig,axes=plt.subplots(2,3,figsize=(12.7,8.3));fig.subplots_adjust(left=.075,right=.97,bottom=.12,top=.85,wspace=.32,hspace=.42)
  for i,ds in enumerate(['hc2','vidstg']):
   rows=[r for r in rr if r['dataset']==ds]
   for j,(a,b) in enumerate([('Source','Inherited_order1'),('Source','Inherited_order2'),('Inherited_order1','Inherited_order2')]):
    ax=axes[i,j];pairs=[(100*r[a+'_own_U_delta'],100*r[b+'_own_U_delta']) for r in rows if r[a+'_own_U_delta'] is not None and r[b+'_own_U_delta'] is not None];x,y=np.array(pairs).T
    colors=['#cf4b42' if u*v<0 and abs(u)>EPS and abs(v)>EPS else '#226e9c' for u,v in pairs];lim=max(5,float(max(np.abs(x).max(),np.abs(y).max()))*1.12)
    ax.scatter(x,y,c=colors,s=27,alpha=.85,edgecolors='white',linewidth=.35);ax.plot([-lim,lim],[-lim,lim],color='#999999',lw=.8);ax.axhline(0,color='#aaaaaa',lw=.6);ax.axvline(0,color='#aaaaaa',lw=.6)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel=a.replace('_',' ')+' utility (pp)',ylabel=b.replace('_',' ')+' utility (pp)');v=s['datasets'][ds]['own_U']['contrasts'][a+'_to_'+b]
    ax.set_title(f'{ds.upper()} | {v["defined_pairs"]} defined parents\n|difference| {fmt(v["absolute_difference"])} pp; flips {v["sign_flips"]}')
  fig.suptitle('Controlled receiving-LN initialization: identical evidence and noise',fontsize=16,y=.965)
  fig.text(.5,.91,'Each point is one historical parent. Utility = After IoU - Before IoU on genuine Omega0-unobserved frames.',ha='center',fontsize=10)
  fig.text(.5,.035,'Red = opposite utility signs. Gray diagonal = unchanged utility. No GT-selected state/step or deployable gate.',ha='center',fontsize=10)
  pdf.savefig(fig);plt.close(fig)
 pdfs.append((p1,1))
 p2=figdir/'CONTROLLED_LN_TAILS_AND_READBACK.pdf'
 with PdfPages(p2) as pdf:
  fig,axes=plt.subplots(2,2,figsize=(11.7,8.3));fig.subplots_adjust(left=.08,right=.97,bottom=.12,top=.84,hspace=.43,wspace=.30)
  colors=['#226e9c','#df8e21','#9b4f96']
  for i,ds in enumerate(['hc2','vidstg']):
   rows=[r for r in rr if r['dataset']==ds]
   for j,a in enumerate(ARMS):
    vals=sorted(100*r[a+'_own_U_delta'] for r in rows if r[a+'_own_U_delta'] is not None);axes[i,0].plot(range(1,len(vals)+1),vals,'.-',ms=4,color=colors[j],label=a.replace('_',' '))
    v=derived[ds][a];axes[i,1].plot([j-.12,j+.12],[100*v['before']['mean'],100*v['after']['mean']],'-o',color=colors[j],label=a.replace('_',' '))
    axes[i,1].text(j-.12,100*v['before']['mean'],'B',ha='center',va='bottom',fontsize=8);axes[i,1].text(j+.12,100*v['after']['mean'],'A',ha='center',va='top',fontsize=8)
   axes[i,0].axhline(0,color='#999999',lw=.7);axes[i,0].axhline(-5,color='#cf4b42',ls='--',lw=.6);axes[i,0].axhline(-20,color='#cf4b42',ls=':',lw=.6)
   axes[i,0].set(title=ds.upper()+' complete utility distributions',xlabel='Parent rank within arm (separately sorted)',ylabel='Update utility (pp)');axes[i,0].legend(fontsize=8)
   axes[i,1].set(title=ds.upper()+' initial versus final genuine-U readout',xticks=range(3),xticklabels=['Source','Inherited 1','Inherited 2'],ylabel='Parent mean IoU (%)');axes[i,1].grid(axis='y',alpha=.15)
  fig.suptitle('Before/After decomposition and all negative tails',fontsize=16,y=.965);fig.text(.5,.91,'32 historical parents per domain. All 21 empty fits and any undefined U remain in the exported tables.',ha='center',fontsize=10)
  fig.text(.5,.035,'A better initial prediction is distinct from a more useful update. Dashed tails are prespecified -5/-20 pp.',ha='center',fontsize=10)
  pdf.savefig(fig);plt.close(fig)
  fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_axes([.08,.1,.84,.79]);ax.axis('off')
  lines=[('Actual qualification', '24 full episodes; 8 fresh captures; exact complete repeats'),('Actual formal', '192 new complete fits; 4800 rounds; 64 frozen captures'),('Original-control replay', '128 complete inherited fits exactly match original Full'),('Qualification-formal replay', '12 complete formal fits exactly match qualification'),('Entering state', 'Only six LN tensors change; Query Residual = 0'),('Optimizer and random numbers', 'Fresh Adam per arm; identical underlying antithetic noise'),('Empty cases', '7 Vid queries x 3 arms = 21 full no-sampler/no-optimizer episodes'),('Teacher and temporal support', 'Exact Omega0 cached pack; no new DINO; Native WHEN fixed'),('GT barrier', 'All 192 predictions sealed before scoring'),('Independent Adam coordinates', str(root['counts']['independent_Adam_scalar_checks'])),('Replayed sample coordinates', str(root['counts']['sample_coordinates'])),('Independent outcome/statistics', str(root['independent_scalar_checks'])),('Inference boundary', 'Actual forward + native-head VJP; no independent full Jacobian'),('Decision boundary', 'Historical-panel causality; no unlabelled gate or promotion')]
  table=ax.table(cellText=lines,colLabels=['Control / verification','Actual scope'],colWidths=[.34,.66],loc='center',cellLoc='left');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.02)
  for (row,col),cell in table.get_celld().items():
   cell.set_edgecolor('#d2dce3');cell.set_facecolor('#e7f0f6' if row==0 else '#f5f8fa' if row%2 else 'white')
   if row==0:cell.set_text_props(weight='bold')
  fig.suptitle('Evidence controls and limits of the intervention',fontsize=16,y=.95);fig.text(.5,.055,'Private states, raw fits, samples, teacher geometry and GT remain local. No cancelled baseline is resumed.',ha='center',fontsize=10);pdf.savefig(fig);plt.close(fig)
 pdfs.append((p2,2))
 write(BASE/'FIGURE_EXPORT.json',dict(status='exported_pending_actual_pixel_view',PDF_files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size,pages=n) for p,n in pdfs},PDFs=2,pages=3,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':run()
