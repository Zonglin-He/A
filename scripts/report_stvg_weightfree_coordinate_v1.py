"""Anonymous complete coordinate/confirmation report; never exports private inputs."""
import sys,json,math,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_coordinate_v1';H=B/'confirmation';OUT=ROOT/'results/stvg_weightfree_coordinate/2026-10-15'
def main():
 import torch
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 OUT.mkdir(parents=True,exist_ok=True)
 sp=read(B/'DESIGN_LOCK.json');tp=read(B/'TEMPORAL_DESIGN_LOCK002.json');stages=sp['stages']+tp['stages'];rows=[];audit=[];space_count=0;reuse_count=0;no_reference_count=0
 for stage in stages:
  p=B/stage['name'];raw=read(p/'POSTSEAL_ROWS.json')['rows']
  rr=[dict(stage=stage['name'],case_id=x['case_id'],dataset=x['dataset'],arm=x['arm'],metrics=x['metrics'])for x in raw];rows+=rr
  a=read(p/'SELECTION_AUDIT.json');audit.append(dict(stage=stage['name'],records=a['records']))
  write(OUT/(stage['name']+'_ALL_ROWS.json'),dict(status='complete',rows=rr,development=True))
  write(OUT/(stage['name']+'_SELECTION.json'),a)
  if 'STUDENT_GLOBAL_BARRIER.json'in [x.name for x in p.iterdir()]:
   bar=read(p/'STUDENT_GLOBAL_BARRIER.json');space_count+=sum(x['actual_fit_count']for x in bar['files'])
   for item in bar['files']:
    d=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
    reuse_count+=sum(r['status'].startswith('exact_')for r in d['receipts'].values());no_reference_count+=sum(r['status']=='no_reference_no_update'for r in d['receipts'].values())
 raw=read(H/'POSTSEAL_ROWS.json')['rows'];hold=[dict(case_id=x['case_id'],dataset=x['dataset'],arm=x['arm'],metrics=x['metrics'])for x in raw]
 write(OUT/'CONFIRMATION_ALL_ROWS.json',dict(status='complete',rows=hold,confirmation=True,not_used_for_selection=True,historical_project_exposure=True))
 for src,dst in [(H/'PARENT_PAIRED_DELTAS.json','CONFIRMATION_PARENT_PAIRED_DELTAS.json'),(B/'DESIGN_LOCK.json','SPATIAL_DESIGN.json'),(B/'TEMPORAL_DESIGN_LOCK002.json','TEMPORAL_DESIGN.json'),(B/'JOINT_METHOD_LOCK.json','JOINT_METHOD_LOCK.json'),(B/'CAUSE_READBACK.json','CAUSE_READBACK.json')]:shutil.copyfile(src,OUT/dst)
 diag=read(B/'temporal_diagnosis/POSTSEAL_ROWS.json');write(OUT/'TEMPORAL_DIAGNOSIS_ALL_ROWS.json',diag)
 tb=read(H/'TEMPORAL_GLOBAL_BARRIER.json');cb=read(H/'CANDIDATE_INPUT_BARRIER.json');teacher=read(H/'TEACHER_GLOBAL_BARRIER.json');refs=read(H/'REFERENCE_LOCK.json')['records'];tr=[read(ROOT/x['path'])for x in teacher['files']]
 costs=dict(status='complete',confirmation_sources=32,temporal_teacher_calls=tb['teacher_calls'],temporal_phase_wall_seconds_excluding_model_load=tb['wall_seconds'],temporal_model_load_seconds=tb['model_load_seconds'],spatial_teacher_calls=len(tr),spatial_teacher_generation_seconds=sum(x['model_forward_generate_seconds']for x in tr),spatial_teacher_input_tokens=sum(x['input_tokens']for x in tr),spatial_teacher_visual_tokens=sum(x['visual_tokens']for x in tr),spatial_teacher_output_tokens=sum(x['output_tokens']for x in tr),proposal_windows=cb['window_count'],proposal_compute_seconds=cb['proposal_seconds'],proposal_phase_wall_seconds=cb['wall_seconds'],reference_sources=sum(bool(x['original']['anchors'])for x in refs),reference_anchor_count=sum(len(x['original']['anchors'])for x in refs),no_reference_sources=sum(not x['original']['anchors']for x in refs),new_detection_segmentation_models=0,GT_oracle_fits=0,not_end_to_end_speedup_estimate=True)
 write(OUT/'CONFIRMATION_COST_COUNTS.json',costs)
 final=read(B/'JOINT_METHOD_LOCK.json');summ=read(H/'POSTSEAL_SUMMARY.json')['summary'];names=['F','T','T_tuned','original','T_original','selected','T_selected','T_tuned_selected'];labels=['Frozen','Previous Temporal','Tuned Temporal','Previous Spatial','Previous T+S','Tuned Spatial','Previous T + tuned S','Tuned T+S']
 means={(x['dataset'],x['arm']):100*x['mean']for x in summ if x['metric']=='m_vIoU'}
 text=['# Sequential parameter tuning and expanded confirmation','', 'Actual scope: 64 parent sources, 32 per direction. The formerly scored32 are now explicitly development; the other32 were excluded from all coordinate selection. Both cohorts have historical project exposure. This is not Table1 or an untouched full-test estimate.','', '## Failure diagnosis','', '28 usable temporal inputs reproduced exactly from frozen features. The original final regularized preference objective exceeded its initial value in19 cases (HC11, Vid8). On the exposed32, Projection-minus-Native mean vIoU was +0.585524/+0.041443pp, while temporal adaptation-minus-Projection was −2.608173/−0.277270pp. HC06 preferred a lower-quality contracted candidate and then decoded an even worse interval; HC14 had both a worse Projection prior and further adaptation harm. These support separate feedback-ranking and update-amplitude explanations, not a proof that one parameter explains every failure.','', 'Four sources improved training pseudo-reference agreement but lost full-tube performance: HC02, HC14, V06, V12. Teacher agreement is not a correctness guarantee.','', '## Sequential search','', 'Every stage inherits the previous selected values, per dataset; all other factors stay fixed. Each stage seals all32 predictions before scoring. Selection uses development GT only. All search points, negative and neutral results are exported. Finite grid winners are not global optima. Spatial fitting always starts from the source checkpoint; exact config/source/reference/seed reuse is separately counted.','', '|Stage|Dataset|Parameter|Selected value|Development vIoU (%)|','|---|---|---|---:|---:|']
 for a in audit:
  for r in a['records']:
   score=r['scores'][r['selected']]*100;text.append(f"|{a['stage']}|{r['dataset']}|{r['parameter']}|{r['config'][r['parameter']]:.8g}|{score:.6f}|")
 text+=['','Spatial stage scores use fixed original Temporal plus varied Spatial; temporal stage scores use the frozen final spatial winner. Both coordinate chains therefore select the same joint T+S endpoint; Temporal-only is an attribution control.','', '## Frozen final configuration','', '```json',json.dumps(dict(spatial=final['spatial'],temporal=final['temporal']),indent=2),'```','', 'The original five-step QA output still fixes spatial observation positions in both cohorts. The selected temporal configuration governs the final interval. This deliberate fixed-evidence protocol does not claim references were regenerated under the tuned interval. A zero-lr winner makes the final-output head Projection-only, not learned temporal adaptation. The original QA head still supplies the fixed spatial observation clock. Once coordinate selection chooses zero lr, later temporal temperature/step/regularization sweeps are algebraically inactive for that domain; flat results do not prove these parameters are intrinsically irrelevant or that no positive-lr joint configuration could work. This is a finite single coordinate pass, not global optimization. Episodic writeback is inactive.','', '## Confirmation (16 parent sources per direction)','', '|Arm|HC vIoU (%)|VidSTG vIoU (%)|','|---|---:|---:|']
 for n,l in zip(names,labels):text.append(f'|{l}|{means["hc2",n]:.6f}|{means["vidstg",n]:.6f}|')
 text+=['','|Dataset|Contrast|Mean delta (pp)|Worst source delta (pp)|Negative sources|','|---|---|---:|---|---:|']
 for r in read(H/'PARENT_PAIRED_DELTAS.json')['records']:
  text.append(f"|{r['dataset']}|{r['left']} − {r['right']}|{r['mean_delta_pp']:.6f}|{r['worst_delta_pp']:.6f}|{r['negative_sources']}|")
 hc=read(H/'STUDENT_GLOBAL_BARRIER.json');new_hold=sum(x['actual_fit_count']for x in hc['files'])
 text+=['','## Actual confirmation workload','',f"Temporal teacher calls: {costs['temporal_teacher_calls']}; spatial teacher calls: {costs['spatial_teacher_calls']}; local proposal windows: {costs['proposal_windows']}; sources with selected references: {costs['reference_sources']}/32, anchors: {costs['reference_anchor_count']}. All counts and measured component costs are exported in CONFIRMATION_COST_COUNTS.json. These are component measurements, not an end-to-end speedup against full-video YOLO/SAM2."]
 text+=['','## Execution and limits','',f'Actual new spatial fits/full Student redecodes: development {space_count}; confirmation {new_hold}. Development exact config/source/reference/seed reuse records: {reuse_count}; no-reference no-update records: {no_reference_count}. Shared temporal-head trials reuse exact frozen backbone features and saved teacher feedback, and are not counted as new full-backbone or MLLM inference. Confirmation reuses source captures only after source/pixel/state checks; fresh teacher calls and input generation are separately recorded. No new detector/segmenter, GT Oracle fit, CURRENT promotion, old queue resume or Table1 launch.','', 'The first launcher failed its own cgroup-name assertion before any GPU fit; a separate002 supervisor fixed initialization ordering. Original capture/source/runtime were retained. Synthetic contracts are not scientific-effect evidence.','', 'Selection bias remains in development scores. Confirmation is not used to restart tuning in this round. Bootstrap is omitted by the latest user instruction; mean, all paired source differences and negative tails are reported. Small parent samples and historical exposure limit generalization claims. Negative outcomes remain in the public rows.']
 report='\n'.join(text)+'\n';dest=ROOT/'docs/STVG_WEIGHTFREE_COORDINATE_REPORT.md';dest.write_text(report);(OUT/'REPORT.md').write_text(report)
 fig,ax=plt.subplots(1,2,figsize=(14,5.3),constrained_layout=True)
 for a,ds,title in zip(ax,['hc2','vidstg'],['HC-STVG-v2','VidSTG']):
  vals=[means[ds,n]for n in names];a.bar(range(len(vals)),vals,color=['#727272','#C59745','#AFC2DF','#B0C5AA','#AC8CB0','#7BA292','#79A8BD','#245985']);a.set_xticks(range(len(vals)),labels,rotation=40,ha='right');a.set_title(title+' — 16 confirmation sources');a.set_ylabel('vIoU (%)');a.grid(axis='y',alpha=.18)
  for i,v in enumerate(vals):a.text(i,v+.15,f'{v:.2f}',ha='center',fontsize=8)
 fig.savefig(OUT/'CONFIRMATION.png',dpi=160);plt.close(fig)
 for category,ss in [('SPATIAL',sp['stages']),('TEMPORAL',tp['stages'])]:
  fig,axes=plt.subplots(len(ss),2,figsize=(12,2.55*len(ss)),squeeze=False,constrained_layout=True)
  for i,stage in enumerate(ss):
   aa=read(B/stage['name']/'SELECTION_AUDIT.json')['records'];recipes=read(B/stage['name']/'STAGE_LOCK.json')['recipes']
   for j,ds in enumerate(['hc2','vidstg']):
    r=next(x for x in aa if x['dataset']==ds);names2=list(r['scores']);xx=[recipes[ds][v][r['parameter']]for v in names2];yy=[100*r['scores'][v]for v in names2];a=axes[i,j];a.plot(range(len(xx)),yy,'o-',color='#336b88');win=names2.index(r['selected']);a.scatter([win],[yy[win]],s=75,color='#ba7037',zorder=3);a.set_xticks(range(len(xx)),[f'{v:.5g}'for v in xx]);a.set_title(ds+' / '+stage['name']);a.set_xlabel(r['parameter']);a.set_ylabel('Development joint vIoU (%)');a.grid(alpha=.2)
  fig.savefig(OUT/(category+'_COORDINATES.png'),dpi=140);plt.close(fig)

 # Independently aggregate every public row and match the private scalar stream exactly.
 assert len(hold)==len(raw)
 for x,y in zip(hold,raw):assert x['metrics']==y['metrics']
 for (ds,arm),v in means.items():
  vals=[x['metrics']['m_vIoU']*100 for x in hold if x['dataset']==ds and x['arm']==arm];assert len(vals)==16 and abs(math.fsum(vals)/16-v)<1e-10
 write(B/'ROOT_ANONYMOUS_SCALAR_READBACK.json',dict(status='pass',development_rows=len(rows),confirmation_rows=len(hold),all_signs_and_nulls_preserved=True,confirmation_mean_reaggregation=True))
 print('report',space_count,new_hold,means)
if __name__=='__main__':main()
