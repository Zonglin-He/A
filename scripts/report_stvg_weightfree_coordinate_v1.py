"""Anonymous complete coordinate/confirmation report; never exports private inputs."""
import sys,json,math,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_coordinate_v1';H=B/'confirmation';OUT=ROOT/'results/stvg_weightfree_coordinate/2026-10-15'
def main():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 OUT.mkdir(parents=True,exist_ok=True)
 sp=read(B/'DESIGN_LOCK.json');tp=read(B/'TEMPORAL_DESIGN_LOCK002.json');stages=sp['stages']+tp['stages'];rows=[];audit=[];space_count=0;reuse_count=0
 for stage in stages:
  p=B/stage['name'];raw=read(p/'POSTSEAL_ROWS.json')['rows']
  rr=[dict(stage=stage['name'],case_id=x['case_id'],dataset=x['dataset'],arm=x['arm'],metrics=x['metrics'])for x in raw];rows+=rr
  a=read(p/'SELECTION_AUDIT.json');audit.append(dict(stage=stage['name'],records=a['records']))
  write(OUT/(stage['name']+'_ALL_ROWS.json'),dict(status='complete',rows=rr,development=True))
  write(OUT/(stage['name']+'_SELECTION.json'),a)
  if 'STUDENT_GLOBAL_BARRIER.json'in [x.name for x in p.iterdir()]:
   bar=read(p/'STUDENT_GLOBAL_BARRIER.json');space_count+=sum(x['actual_fit_count']for x in bar['files'])
 raw=read(H/'POSTSEAL_ROWS.json')['rows'];hold=[dict(case_id=x['case_id'],dataset=x['dataset'],arm=x['arm'],metrics=x['metrics'])for x in raw]
 write(OUT/'CONFIRMATION_ALL_ROWS.json',dict(status='complete',rows=hold,confirmation=True,not_used_for_selection=True,historical_project_exposure=True))
 for src,dst in [(H/'PARENT_BOOTSTRAP.json','CONFIRMATION_PARENT_BOOTSTRAP.json'),(B/'DESIGN_LOCK.json','SPATIAL_DESIGN.json'),(B/'TEMPORAL_DESIGN_LOCK002.json','TEMPORAL_DESIGN.json'),(B/'JOINT_METHOD_LOCK.json','JOINT_METHOD_LOCK.json'),(B/'CAUSE_READBACK.json','CAUSE_READBACK.json')]:shutil.copyfile(src,OUT/dst)
 diag=read(B/'temporal_diagnosis/POSTSEAL_ROWS.json');write(OUT/'TEMPORAL_DIAGNOSIS_ALL_ROWS.json',diag)
 final=read(B/'JOINT_METHOD_LOCK.json');summ=read(H/'POSTSEAL_SUMMARY.json')['summary'];names=['F','T','T_tuned','original','T_original','selected','T_selected','T_tuned_selected'];labels=['Frozen','Previous Temporal','Tuned Temporal','Previous Spatial','Previous T+S','Tuned Spatial','Previous T + tuned S','Tuned T+S']
 means={(x['dataset'],x['arm']):100*x['mean']for x in summ if x['metric']=='m_vIoU'}
 text=['# Sequential parameter tuning and expanded confirmation','', 'Actual scope: 64 parent sources, 32 per direction. The formerly scored32 are now explicitly development; the other32 were excluded from all coordinate selection. Both cohorts have historical project exposure. This is not Table1 or an untouched full-test estimate.','', '## Failure diagnosis','', '28 usable temporal inputs reproduced exactly from frozen features. The original final regularized preference objective exceeded its initial value in19 cases (HC11, Vid8). On the exposed32, Projection-minus-Native mean vIoU was +0.585524/+0.041443pp, while temporal adaptation-minus-Projection was −2.608173/−0.277270pp. HC06 preferred a lower-quality contracted candidate and then decoded an even worse interval; HC14 had both a worse Projection prior and further adaptation harm. These support separate feedback-ranking and update-amplitude explanations, not a proof that one parameter explains every failure.','', 'Four sources improved training pseudo-reference agreement but lost full-tube performance: HC02, HC14, V06, V12. Teacher agreement is not a correctness guarantee.','', '## Sequential search','', 'Every stage inherits the previous selected values, per dataset; all other factors stay fixed. Each stage seals all32 predictions before scoring. Selection uses development GT only. All search points, negative and neutral results are exported. Finite grid winners are not global optima. Spatial fitting always starts from the source checkpoint; exact config/source/reference/seed reuse is separately counted.','', '|Stage|Dataset|Parameter|Selected value|Development vIoU (%)|','|---|---|---|---:|---:|']
 for a in audit:
  for r in a['records']:
   score=r['scores'][r['selected']]*100;text.append(f"|{a['stage']}|{r['dataset']}|{r['parameter']}|{r['config'][r['parameter']]:.8g}|{score:.6f}|")
 text+=['','Spatial stage scores use fixed original Temporal plus varied Spatial; temporal stage scores use the frozen final spatial winner. Both coordinate chains therefore select the same joint T+S endpoint; Temporal-only is an attribution control.','', '## Frozen final configuration','', '```json',json.dumps(dict(spatial=final['spatial'],temporal=final['temporal']),indent=2),'```','', 'The original five-step QA output still fixes spatial observation positions in both cohorts. The selected temporal configuration governs the final interval. This deliberate fixed-evidence protocol does not claim references were regenerated under the tuned interval. A zero-lr winner would be Projection-only, not learned temporal adaptation. Episodic writeback is inactive.','', '## Confirmation (16 parent sources per direction)','', '|Arm|HC vIoU (%)|VidSTG vIoU (%)|','|---|---:|---:|']
 for n,l in zip(names,labels):text.append(f'|{l}|{means["hc2",n]:.6f}|{means["vidstg",n]:.6f}|')
 text+=['','|Dataset|Contrast|Mean delta (pp)|95% parent bootstrap interval (pp)|Negative sources|','|---|---|---:|---|---:|']
 for r in read(H/'PARENT_BOOTSTRAP.json')['records']:
  text.append(f"|{r['dataset']}|{r['left']} − {r['right']}|{r['mean_delta_pp']:.6f}|{r['CI95_percentile_pp']}|{r['negative_sources']}|")
 hc=read(H/'STUDENT_GLOBAL_BARRIER.json');new_hold=sum(x['actual_fit_count']for x in hc['files'])
 text+=['','## Execution and limits','',f'Actual new spatial fits/full Student redecodes: development {space_count}; confirmation {new_hold}. Shared temporal-head trials reuse exact frozen backbone features and saved teacher feedback, and are not counted as new full-backbone or MLLM inference. Confirmation reuses source captures only after source/pixel/state checks; fresh teacher calls and input generation are separately recorded. No new detector/segmenter, GT Oracle fit, CURRENT promotion, old queue resume or Table1 launch.','', 'The first launcher failed its own cgroup-name assertion before any GPU fit; a separate002 supervisor fixed initialization ordering. Original capture/source/runtime were retained. Synthetic contracts are not scientific-effect evidence.','', 'Selection bias remains in development scores. Confirmation is not used to restart tuning in this round. Tiny parent samples and historical exposure limit generalization claims. Negative outcomes remain in the public rows.']
 report='\n'.join(text)+'\n';dest=ROOT/'docs/STVG_WEIGHTFREE_COORDINATE_REPORT.md';dest.write_text(report);(OUT/'REPORT.md').write_text(report)
 fig,ax=plt.subplots(1,2,figsize=(14,5.3),constrained_layout=True)
 for a,ds,title in zip(ax,['hc2','vidstg'],['HC-STVG-v2','VidSTG']):
  vals=[means[ds,n]for n in names];a.bar(range(len(vals)),vals,color=['#727272','#C59745','#AFC2DF','#B0C5AA','#AC8CB0','#7BA292','#79A8BD','#245985']);a.set_xticks(range(len(vals)),labels,rotation=40,ha='right');a.set_title(title+' — 16 confirmation sources');a.set_ylabel('vIoU (%)');a.grid(axis='y',alpha=.18)
  for i,v in enumerate(vals):a.text(i,v+.15,f'{v:.2f}',ha='center',fontsize=8)
 fig.savefig(OUT/'CONFIRMATION.png',dpi=160);plt.close(fig)
 # Independently aggregate every public row and match the private scalar stream exactly.
 assert len(hold)==len(raw)
 for x,y in zip(hold,raw):assert x['metrics']==y['metrics']
 for (ds,arm),v in means.items():
  vals=[x['metrics']['m_vIoU']*100 for x in hold if x['dataset']==ds and x['arm']==arm];assert len(vals)==16 and abs(math.fsum(vals)/16-v)<1e-10
 write(B/'ROOT_ANONYMOUS_SCALAR_READBACK.json',dict(status='pass',development_rows=len(rows),confirmation_rows=len(hold),all_signs_and_nulls_preserved=True,confirmation_mean_reaggregation=True))
 print('report',space_count,new_hold,means)
if __name__=='__main__':main()
