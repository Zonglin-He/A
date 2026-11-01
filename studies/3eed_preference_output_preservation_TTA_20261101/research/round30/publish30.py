import sys,io,csv,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round30'))
from common30 import *
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE')}
 if isinstance(x,list):return [clean(v) for v in x]
 return x

def main():
 assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED';pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(src,dst):
  p=out/dst;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
 for folder in [BASE,EXP.parent]:
  for p in sorted(folder.glob('*.py')):cp(p,p.relative_to(ROOT))
 cp(EXP/'REPORT.md','REPORT.md')
 for n in ['PLAN_LOCK','MEMORY_KEEP_CONTROL_LOCK','SELECTED_BETA_LOCK','S64_EXPOSURE_LOCK','FINAL_CODE_SNAPSHOT']:cp(EXP/(n+'.json'),Path('protocol')/(n+'.json'))
 for n in ['CPU_AUDIT','DECISION','DEV64_ALL_OPTIONS','TASK_SUMMARY','A_FEEDBACK_STRATA','SCENE_CLUSTER_COMPARISONS','S64_SCENE_RESULTS','PAIR_EXECUTION_SUMMARY','THIRD_CANDIDATE_GRAPH_ROWS','THIRD_CANDIDATE_GRAPH_SUMMARY','EVIDENCE22_TYPE_COUNTS','EVIDENCE22_EXECUTION_ROWS','EVIDENCE22_STRATA_SUMMARY','ALL_NEGATIVES']:cp(EXP/(n+'.json'),Path('results')/(n+'.json'))
 cp(BASE/'screen_attempt001/FIT_RECEIPTS.json','results/ALL8_FIT_RECEIPTS.json');cp(BASE/'closureS64_attempt001/CLOSURE_AUDIT.json','results/FULLMODEL_CLOSURE_AUDIT.json');cp(BASE/'evidence_review22_attempt001/ANNOTATIONS.json','review/AVAILABLE_EVIDENCE22_ANNOTATIONS.json');cp(BASE/'evidence_review22_attempt001/SCOPE.json','review/SCOPE.json');dump(out/'results/PAIR_EXECUTION_ROWS.json',clean(read(EXP/'PAIR_EXECUTION_ROWS_PRIVATE.json')));dump(out/'results/DEV64_ALL_ROWS.json',clean(read(EXP/'DEV64_ROWS_PRIVATE.json')))
 rr=read(EXP/'ALL336_TASK_ROWS_PRIVATE.json');scenes={p:sorted({r['scene_PRIVATE'] for r in rr if r['platform']==p}) for p in ['drone','quad']};buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['split','alias','platform','scene_alias','arm','iou','query_changed_vs_source','has_accepted_feedback'],lineterminator='\n');w.writeheader();n=0
 for r in rr:
  for a,v in r['ious'].items():w.writerow({'split':r['split'],'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_s'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'arm':a,'iou':v,'query_changed_vs_source':r['q'][a]!=r['q']['source'],'has_accepted_feedback':r['has_accepted_feedback']});n+=1
 (out/'results/ALL336_SCALAR_ROWS.csv').write_text(buf.getvalue())
 (out/'README.md').write_text('''# R30 — preserve unlabelled outputs during matching preference adaptation

Read REPORT.md. Same22R29external pseudo preferences, sameWaymo-onlysource,
matchingqueryMLP184960params and logodds RankNet. Single mainchange: relative
sourceq0 logodds output-MSE on189Aobservations withoutacceptedfeedback.
Finitebeta[0,.1,1,10],AdamW5e-4x16/platform; beta0 exactR29state replay.
One shared beta1 chosen on exposedoldQ64 withprespecified correctivefeasibility.
R29R64never used for choosingbeta. NewS64 contentdisjointprior592+declaredhistory,
but2Drone/3Quad historicalscenes andpriorfullVALlabel exposure; Snotfit/Ginput.

Keep reducesDroneU harm22->12 and retains6/6requestedreplacement pseudopreferences
(includingoneknownbadfeedback), not merely frozenrecovery. S64Drone-1.653647pp,
Quad+2.839300pp,balanced+.592827pp vs source; sceneCI crosseszero. Not adopted as
robustcrossplatform method. ActiveA-memory andallAzeroU-memory no-update controls
retained. No platformwiseposthoc hybrid or Sdriven retuning.

Pairgraph explains one apparentthirdcandidate as anotheracceptedpositive; one
uncoveredthird remains. No automatic negatives/newQwen comparisons. Available
feedbackevidence19mixed,1appearance-only,2neither(sameobservation), no certified
puregeometry group. Not proof ofQwenreasoning orXYZobservability, not an appearance
versusgeometry causaltransfer comparison. Badfeedback unchanged, no manualfilter.

CPUaudit reconstructs8keepobjectives, selection,1344taskscalars,44pairgraphstates,
246negative comparisons;128fullmodelclosureforwards exactargmax/frozenboxes and
unchangednonmatchingweights. NewQwencalls0, newsourcefulltraining0. SourceWaymo
forgetting was not evaluated. EATA/EcoTTA/newRGBmodule not introduced.

Privateassets andprior studies through0e7503a required. No rawRGB/pointclouds,
captions, opaqueIDs, coordinates, features orweightsredistributed. Usefreshoutput
attemptdirs; originalseals never overwrite. Workflow:fit30 -> select30 ->
select_new30/capture30 -> predict30 andmemory_all30 -> closure30 -> seal_evidence30
-> score30 -> mechanism30/graph30 -> audit30/audit_graph30 -> finalize30/publish30.
Allscalararms/negative comparisons preserved; repeatedpairs/views are not
independent observations. Seeprotocol files for selection andsourcecontracts.
''')
 dump(out/'DEPENDENCIES.json',{'parent_commit':'0e7503aea1eeda4e2d44840c3f6296902aa9fa80','student_input':'XYZ+text','G_input':'cached external frozenQwen RGB+fulltext','Qwen_new_calls':0,'thirdparty_data_weights_redistributed':False,'LwF':'inspiration, not formula reproduction'})
 forbidden={str(r[k]) for r in prior.rows()+read(ROOT/'research/round29/input/R64_PRIVATE.json')+rows('S64') for k in ['id','caption','imagepath','pointpath','metapath','scene']};files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 for p in files:assert not any(t and t in p.read_text() for t in forbidden),p
 dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True});files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_preference_output_preservation_TTA_20261101';dump(pub/'TREE_PAYLOAD_LOCAL.json',[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files]);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'scalar_rows':n,'all_negative_comparisons':246,'private_payload_scan_passed':True,'prefix':prefix});print('EXPORT30',len(files),n)
if __name__=='__main__':main()
