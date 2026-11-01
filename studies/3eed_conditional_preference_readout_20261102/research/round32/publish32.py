"""Local, allowlisted R32 export. No GPU, scoring, network write or reserve access."""
import sys,io,csv,shutil,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round32'))
from common32 import read,dump,sha,BASE,EXP,rows

def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE')}
 if isinstance(x,list):return [clean(v) for v in x]
 return x

def main():
 ia=EXP.parent/'independent_audit';audit=read(ia/'AUDIT_RESULT.json');assert audit['state']=='PASS_PRIMARY_SELECTION_TASK_CONTENT_WITH_LEGACY_NUMERICAL_WARNINGS';assert read(EXP/'DECISION.json')['run']=='completed'
 pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(src,dst):
  p=out/dst;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
 def write_csv(dst,records,fields=None):
  assert records;buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields or list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records);p=out/dst;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(buf.getvalue())
 for folder in [BASE,EXP.parent,ia]:
  for p in sorted(folder.glob('*.py')):cp(p,p.relative_to(ROOT))
 cp(EXP/'REPORT.md','REPORT.md')
 for n in ['PLAN_LOCK','ENDPOINT_SCOPE_AMENDMENT_LOCK','SELECTED_SCOPE_LOCK','INTERVAL_MEASUREMENT_NOTE','FINAL_CODE_SNAPSHOT']:cp(EXP/(n+'.json'),Path('protocol')/(n+'.json'))
 for n in ['DECISION','DEV_FAMILY_TABLE','TASK_SUMMARY','ALL_NEGATIVES','T128_SCENE_COMPARISONS','T128_SCENE_RESULTS','PRIMARY_T128_SCOPE_COVERAGE','INTERVAL_INTERPRETATION']:cp(EXP/(n+'.json'),Path('results')/(n+'.json'))
 audit_names=['AUDIT_RESULT','CONTENT_AUDIT_RESULT','READOUT_RECONSTRUCTION_RESULT','SELECTION_TASK_AUDIT_RESULT','SOURCE_HEAD_AUDIT_RESULT','LEGACY_PRECISION_AUDIT_RESULT','GROUP_QP_AUDIT_ROWS','CPU_GPU_GATE_AGGREGATE_DIFFERENCES','CPU_GPU_T_READOUT_DIFFERENCES','FEASIBILITY_CPU_WARNINGS','SCENE_UNCERTAINTY_AUDIT','DEV_FAMILY_TABLE_INDEPENDENT','SELECTED_SCOPE_INDEPENDENT','KNN_NEAR_TIE_DIAGNOSTICS']
 for n in audit_names:dump(out/'audit'/(n+'.json'),clean(read(ia/(n+'.json'))))
 cp(ia/'REPORT.md','audit/REPORT.md');cp(ia/'initial_auditor_reconstruction/READOUT_RECONSTRUCTION_RESULT.json','audit/initial_reconstruction/READOUT_RECONSTRUCTION_RESULT.json')
 initial_log=(ia/'initial_auditor_reconstruction/readout_audit.log').read_text().replace(str(ROOT),'WORKSPACE_ROOT');(out/'audit/initial_reconstruction/readout_audit.log').write_text(initial_log)
 cp(BASE/'closureT128_attempt001/CLOSURE_AUDIT.json','audit/OFFICIAL_GPU_CLOSURE_AUDIT.json')
 # Only metadata from seals; private hash allowlists and prediction arrays are excluded.
 runtime=[]
 for name in ['scope400_attempt001','scope400_endpoint_attempt001','sourceT128_attempt001','selectedT128_attempt001','closureT128_attempt001']:
  x=read(BASE/name/'SEALED.json');runtime.append({'attempt':name,**{k:v for k,v in x.items() if k!='files'}})
 dump(out/'results/RUNTIME_RECEIPTS.json',runtime)
 cp(BASE/'feasibilityA_attempt001/PAIR_EXECUTION.json','results/ALL_A_FEASIBILITY_PAIR_DECISIONS.json')
 dump(out/'results/DEV_ALL_FAMILY_ROWS.json',clean(read(EXP/'DEV_FAMILY_ROWS_PRIVATE.json')))
 rr=read(EXP/'ALL528_TASK_ROWS_PRIVATE.json');assert len(rr)==528
 histories={p:sorted({r['scene_PRIVATE'] for r in rr if r['platform']==p}) for p in ['drone','quad']};per_split={(split,p):sorted({r['scene_PRIVATE'] for r in rr if r['split']==split and r['platform']==p}) for split in ['A208','DEV64','R64','S64','T128'] for p in ['drone','quad']};scalar=[]
 for r in rr:
  for arm,v in r['ious'].items():scalar.append({'split':r['split'],'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_s'+str(per_split[r['split'],r['platform']].index(r['scene_PRIVATE'])),'history_scene_alias':r['platform']+'_all_s'+str(histories[r['platform']].index(r['scene_PRIVATE'])),'arm':arm,'iou':v,'query_changed_vs_source':r['q'][arm]!=r['q']['source'],'query_changed_vs_R30':r['q'][arm]!=r['q']['R30_ungated']})
 assert len(scalar)==6272;write_csv('results/ALL528_SELECTED_TASK_SCALARS.csv',scalar);T=[r for r in scalar if r['split']=='T128'];assert len(T)==1664;write_csv('results/T128_ALL13_TASK_SCALARS.csv',T)
 gate=[]
 for r in read(BASE/'selectedT128_attempt001/PROCESS_STATS.json'):
  for st in r['selected_gate_stats']:gate.append({'alias':r['alias'],'platform':r['platform'],**st})
 write_csv('results/T128_SELECTED_GATE_STATISTICS.csv',gate)
 # Saved posthoc numerical-control scores only. No GT opened or score recomputed here.
 legacy=read(ia/'LEGACY_PRECISION_TASK_ROWS_PRIVATE.json');assert len(legacy)==256;legacy_scalar=[]
 for r in legacy:
  for path,v in r['IoUs'].items():legacy_scalar.append({'alias':r['alias'],'platform':r['platform'],'memory_arm':r['arm'],'arithmetic_path':path,'iou':v,'query_changed_vs_original_GPU':r['queries_PRIVATE'][path]!=r['queries_PRIVATE']['original_GPU'],'delta_vs_original_GPU_pp':100*(v-r['IoUs']['original_GPU'])})
 assert len(legacy_scalar)==1024;write_csv('results/ALL_LEGACY_NUMERICAL_CONTROL_TASK_SCALARS.csv',legacy_scalar)
 diagnostics=[]
 for r in read(ia/'legacy_precision_attempt001/PRECISION_COMPARISON_PRIVATE.json'):
  diagnostics.append({'alias':r['alias'],'platform':r['platform'],'memory_arm':r['arm'],'numpy32_query_changed':r['numpy32_q_PRIVATE']!=r['original_GPU_q_PRIVATE'],'torchCPU32_query_changed':r['torchCPU32_q_PRIVATE']!=r['original_GPU_q_PRIVATE'],'float64_query_changed':r['float64_q_PRIVATE']!=r['original_GPU_q_PRIVATE'],'numpy32_best_minus_GPUchosen':r['numpy32_max_minus_GPUchosen'],'torchCPU32_best_minus_GPUchosen':r['torchCPU32_max_minus_GPUchosen'],'float64_best_minus_GPUchosen':r['float64_max_minus_GPUchosen'],'actual_GPU_neighbors_logged':False})
 write_csv('audit/ALL256_LEGACY_NUMERICAL_COMPARISONS.csv',diagnostics)
 # Numerical contract recorded without sharing its private queries/scores/feature caches.
 precision_seal=read(ia/'legacy_precision_attempt001/SEALED.json');dump(out/'audit/LEGACY_POSTHOC_CONTROL_SEAL_METADATA.json',{k:v for k,v in precision_seal.items() if k!='files'})
 parent=read(ROOT/'research/round31/publication/RECEIPT_FINAL.json')['commit']
 dump(out/'DEPENDENCIES.json',{'prior_study_commit':parent,'R30_head_study_commit':'ccff83b5383aca03fbb65c879ccb27dd7d06f4d4','student_input':'XYZ+text','feedback':'R29cachedQwenRGB+fulltext,all22unchanged','neural_training_this_round':False,'thirdparty_assets_weights_or_source_snapshots_redistributed':False,'GRACE':'locality inspiration, not reproduction','reserved_U_FINAL_row_manifests_media_GT_not_read_by_this_export':True})
 (out/'README.md').write_text('''# R32 — conditional readout of reused R30 preference-adapted heads

Read REPORT.md and audit/REPORT.md. This is a source-support-localized readout,
NOT new neural training, not GRACE reproduction. Same22external pseudo preferences,
source/R30heads, frozenboxes and featurekeys. Endpoint support marks accepted
positive/negative groups, not every query in a feedback row. Old row support
is a separately declared ablation. No inferred negative labels or GT filtering.

Four primary endpoint key/k families on exposedoldQ64 DEV. Primary noun/k1 fixed
beforeT128 labels. R/S are historicaldiagnostic; design was informed byR31, so
not a never-target-label-exposed development history. T128actualRGB/XYZ disjoint
prior656+declaredhistory, but1Drone/3Quad historicalscenes and fullVALexposure.
U128/FINAL384 not methodselection/scoring data. This export never opens their
row manifests/media/GT. Shared scenes remain traceable via history_scene_alias;
scene_alias is per split and matches per-split scene result tables.

T128 scoped-minus-source Drone+.514110pp, Quad0,balanced+.257055pp; same-gate
nonparametric group-QP and global mean/norm amount controls retained. Small gains
from oneDroneeffectcluster do not establish broad crossscene superiority. Every
population sceneCI is unavailable with oneDrone scene. Degenerate [.257055,.257055]
resampling quantiles are not significance; other nondegenerate quantiles remain
conditional diagnostics, not populationCIs. All13Tarms,6272taskscalars and783
negative method/reference rows are present. No method adopted.

Readout uses ell_eps(p)=log(p+1e-6)-log(1+1e-6-p) on saved officialfloat32probabilities,
strictly preserving their saturationties, NOT exact differentiable R30 losslogodds.
Bothheads/sourceq0 anchored; binary endpoints restored exactly. Samegate QP
projects instance group coordinates once each (<=2pairconstraints), storesRAW
shifts and anchors retrieved shifts at currentq0; unmentioned queries remainfree.

IndependentCPU primary/samegateQP T scorevectors match exactly. DEV selection,
actualcontentdisjointness and all6272taskvalues audited. Official GPU closure
ran256fullmodel forwards and1664exactreadoutargmax matches; this is separate from
CPUlegacycontrol replay. Boxes/nonmatchingweights unchanged, source restored.

LegacyR30feedbackA memory float32 not bitportable: literalNumPy32 differs7queries,
TorchCPU32 differs4; rawGPU neighborindices absent. Row/noun/k4 historicalgate
fractions differ3rows by1/256, without DEV score/selection impact. Do not hide
these warnings or claim all13CPU readout queries reproduce exactly. Original
GPU andprimary predictions unchanged. Fixedfloat64 numerical path is posthoc,
not performance-selected or independentconfirmation. All256observationxmemory
rows across4paths (1024taskscalars) retained; initial strict audit failure preserved.

No rawRGB/captions/pointclouds/coordinates/opaqueIDs/features/weights or thirdparty
source snapshots exported. Requires private licensed assets and priorstudies.
Ownscripts preserved at workspace-relativepaths; use freshattemptdirs and do not
overwrite seals. R32 has0new optimizer steps,0Qwen calls. Runtime receipts include
multiplecontrols/closure and are not optimizedsinglemethoddeployment latency.
''')
 # Only previously used manifests; reserve U/FINAL are intentionally not opened.
 history=read(ROOT/'research/round27/input/ALL528_PRIVATE.json')+rows('R64')+rows('S64')+rows('T128');assert len(history)==784;forbidden={str(r[k]) for r in history for k in ['id','caption','imagepath','pointpath','metapath','scene']};files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 for p in files:assert not any(x and x in p.read_text() for x in forbidden),p
 dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True,'reserved_data_not_read':True});files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_conditional_preference_readout_20261102';dump(pub/'TREE_PAYLOAD_LOCAL.json',[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files]);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'all_selected_task_scalar_rows':6272,'T128_13arm_scalar_rows':1664,'legacy_precision_task_scalar_rows':1024,'negative_comparisons':783,'all_signs_and_original_numerical_warnings_retained':True,'private_scan_history_observations':784,'private_payload_scan_passed':True,'no_reserve_manifest_media_or_GT_access':True,'prefix':prefix});print('EXPORT32',len(files),6272,1664,1024)
if __name__=='__main__':main()
