import sys,io,csv,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round29'))
from common29 import *
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE')}
 if isinstance(x,list):return [clean(v) for v in x]
 return x

def main():
 assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED';pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(src,dest):
  p=out/dest;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
 for folder in [BASE,EXP.parent]:
  for p in sorted(folder.glob('*.py')):cp(p,p.relative_to(ROOT))
 cp(EXP/'REPORT.md','REPORT.md')
 for n in ['PLAN_LOCK','FULLSCENE_LOCK','TTA_PILOT_LOCK','READOUT_REPAIR_LOCK','R64_EXPOSURE_LOCK','FINAL_CODE_SNAPSHOT']:cp(EXP/(n+'.json'),Path('protocol')/(n+'.json'))
 for n in ['CPU_AUDIT','DECISION','GATE154_SUMMARY','BLIND24_SUMMARY','BLIND24_UNBLINDED_ROWS','FULLPAIR_BLIND24_ROWS','FULLPAIR_BLIND24_SUMMARY','CORRECTION6_VISUAL_ROWS','CORRECTION6_VISUAL_SUMMARY','DIRECT208_SUMMARY','TASK_SUMMARY','BALANCED_SUMMARY','SCENE_CLUSTER_COMPARISONS','PAIR_EXECUTION_SUMMARY','R64_SCENE_RESULTS']:cp(EXP/(n+'.json'),Path('results')/(n+'.json'))
 for folder,label in [('ranknet_attempt001','PROBABILITY'),('ranknet_logodds_attempt001','LOGODDS')]:cp(BASE/folder/'FIT_RECEIPTS.json',Path('results')/(label+'_FIT_RECEIPTS.json'))
 cp(BASE/'closureR64_attempt001/CLOSURE_AUDIT.json','results/FULL_MODEL_CLOSURE_AUDIT.json')
 cp(BASE/'blind_review24_attempt001/ANNOTATIONS.json','review/BLIND24_ASSISTANT_ANNOTATIONS.json');cp(BASE/'blind_review24_attempt001/SCOPE.json','review/SCOPE.json');cp(BASE/'correction_review6_attempt001/ANNOTATIONS.json','review/POSTFIT_ALL6_CORRECTION_ANNOTATIONS.json')
 for src,dest in [(EXP/'PAIR154_ROWS_PRIVATE.json','PAIR154_POLICY_ROWS.json'),(BASE/'direct208_attempt001/PAIR_PREFERENCES_PRIVATE.json','FULLPAIR154_PREFERENCES.json'),(EXP/'PAIR_EXECUTION_ROWS_PRIVATE.json','PAIR_EXECUTION_ROWS.json')]:dump(out/'results'/dest,clean(read(src)))
 records=read(EXP/'ALL272_TASK_ROWS_PRIVATE.json');scenes={p:sorted({r['scene_PRIVATE'] for r in records if r['platform']==p}) for p in ['drone','quad']};b=io.StringIO();w=csv.DictWriter(b,fieldnames=['split','alias','platform','scene_alias','arm','iou','query_changed_vs_source'],lineterminator='\n');w.writeheader();n=0
 for r in records:
  for arm,v in r['ious'].items():w.writerow({'split':r['split'],'alias':r['alias'],'platform':r['platform'],'scene_alias':f"{r['platform']}_s{scenes[r['platform']].index(r['scene_PRIVATE'])}",'arm':arm,'iou':v,'query_changed_vs_source':r['q'][arm]!=r['q']['source']});n+=1
 (out/'results/ALL272_SCALAR_ROWS.csv').write_text(b.getvalue());cp(EXP/'ALL_NEGATIVES.json','results/ALL_TASK_NEGATIVES.json')
 (out/'README.md').write_text('''# R29 — instance preference gates and frozen-box matching TTA

Read REPORT.md. One fixed full-scene/fulltext pair G,308numeric Qwen forwards;
R22cached154physical pairs gate decomposition;24input-selected blind assistant
reviews plus all6accepted corrections reviewed after fit (29uniqueobservations).
Not expert identity gold. No manual/GT filtering of runtime pseudo feedback.

Waymo-only epoch100 source. Fit on208exposed target observations,22accepted
unique-instance-group pseudo preferences from19observations, separately perplatform.
Full matchingqueryMLP184960params,AdamW5e-4x16,boxes/backbone/decoder frozen.
The probability loss failed to implement6corrective preferences. BeforeBscoring,
one exact monotonic logodds loss-readout repair implemented22/22pairs; all failed
arms retained. Local pair satisfaction does not ensure global Top1 or transfer.

New64B:32/platform, actualdecodedRGB/XYZ disjoint prior528anddeclared history,
but2Drone/3Quad scenes overlap prior data and fullVAL has historical label exposure.
Bnot used for Gconstruction/fitting; predictions sealed before current tasklabels.
Same-A nonparametric cache is the matched transfer control; directsameGno-update
is measured onA. No claim these distinct protocols are interchangeable.

Blogodds Drone-6.452706pp,Quad+4.795222pp,mean-.828742pp; not adopted.
Quad3improved0harmed,Drone5improved11harmed. All negative scalar records included.
128fullmodel adapted-state closure forwards confirm frozenboxes and unchanged
nonmatching weights. CPUaudit redoes1712taskscalars,154raw preference policies,
208group partitions and contentdedup. Sceneclusters, not frames, used for intervals.

Requires private licensed3EED assets and dependencies from parent22e7ef1. No weights,
featurecaches, rawmedia/captions, calibration coordinates or opaque originalIDs
redistributed. Programs preserve all localseals and should use freshattemptdirs.
Workflow: audit_pairs29 -> prepare_review29 -> seal_review29 -> fullpair29 + groups29
-> evaluate_G29 -> select_new29 -> capture29 -> fit29(probability) -> registered
readoutrepair -> fit29 --logodds -> closure29 -> score29 -> mechanism29 -> postfit
correctionreview -> audit29 -> finalize29 -> publish29. Scalar output includes
allarms; repeated arms/views/pairs are not independent observations.
''')
 dump(out/'DEPENDENCIES.json',{'parent_commit':'22e7ef1d86b4132a3f172200a8076bd45f2812d7','original_grounder_input':'XYZ+text','G_extra_input':'RGB+fulltext frozenQwen','supervision':'external VLM pseudo preferences, not teacherfree','thirdparty_weights_or_data_redistributed':False})
 forbidden={str(r[k]) for r in rows()+read(EXP.parent/'input/R64_PRIVATE.json') for k in ['id','caption','imagepath','pointpath','metapath','scene']}
 files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 for p in files:assert not any(x and x in p.read_text() for x in forbidden),p
 dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True});files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_instance_preference_matching_TTA_20261101';dump(pub/'TREE_PAYLOAD_LOCAL.json',[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files]);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'scalar_rows':n,'private_payload_scan_passed':True,'all_negative_records_retained':True,'prefix':prefix});print('EXPORT29',len(files),n)
if __name__=='__main__':main()
