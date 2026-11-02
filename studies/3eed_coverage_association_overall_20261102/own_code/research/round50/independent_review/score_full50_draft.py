"""UNEXECUTED full5912 three-arm scorer; requires independent audit and GT grant."""
from pathlib import Path
import sys,json,hashlib,time,itertools,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/round50/experiment';OUT=ROOT/'research/round50/independent_score/full5912_attempt001'
ARMS=['source','old_G','wideentry_oldW']
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import candidate_ious,transform_gt
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 start=time.perf_counter();auth=read(EXP/'ROOT_SINGLE_FULL_SCORING_AUTHORIZATION.json');planpath=EXP/'FULL_SCORING_PLAN_LOCK.json';plan=read(planpath)
 assert auth['state']=='ROOT_AUTHORIZED_R50_SINGLE_POSTSEAL_FULL5912_TASK_SCORE' and auth['program_sha256']==sha(__file__) and auth['scoring_plan_sha256']==sha(planpath)
 assert plan['n']==5912 and plan['arms']==ARMS and plan['source_manifest']=='research/round8/experiment/full5912_gpu_inputs.json'
 for rel,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert 'round46' not in rel and 'FINAL' not in rel and sha(ROOT/rel)==h
 pred=ROOT/plan['prediction_directory'];assert pred.is_relative_to(ROOT/'crossplatform/round50') and auth['prediction_seal_sha256']==sha(pred/'SEALED.json')
 audit=ROOT/plan['independent_prediction_audit'];assert auth['independent_prediction_audit_sha256']==sha(audit) and read(audit)['state'].startswith('PASS_')
 seal=read(pred/'SEALED.json');assert seal['n']==5912 and seal['arms']==ARMS
 for rel,h in seal['files'].items():assert sha(pred/rel)==h
 schema=plan['prediction_schema'];assert schema=={'arm_names':'arm_names','scores':'scores','queries':'queries','selected_boxes':'selected_original_boxes','source_boxes':'source_boxes','source_scores':'source_scores'}
 assert not OUT.exists();OUT.mkdir(parents=True)
 # This is the first decode of the full evaluation input/metadata by this
 # program, strictly after the grant and immutable prediction/audit checks.
 rows=read(ROOT/plan['source_manifest']);assert len(rows)==5912 and len({r['id'] for r in rows})==5912 and len({(r['imagepath'],r['pointpath'],r['metapath']) for r in rows})==5912
 assert collections.Counter(r['platform'] for r in rows)=={'drone':2984,'quad':2928};byid={r['id']:r for r in rows}
 dev=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');identity=lambda r:tuple(r[k] for k in ('imagepath','pointpath','metapath','caption'))
 dev_by_identity={identity(r):r for r in dev};full_by_identity={identity(r):r for r in rows};assert len(dev_by_identity)==336 and len(full_by_identity)==5912
 devmap={};seed_equal=0
 for key,oldrow in dev_by_identity.items():
  assert key in full_by_identity;fullrow=full_by_identity[key]
  assert fullrow['platform']==oldrow['platform'] and fullrow['scene']==oldrow['scene']
  devmap[fullrow['id']]=oldrow;seed_equal+=fullrow.get('seed')==oldrow.get('seed')
 assert len(devmap)==336
 seed_mismatch=336-seed_equal
 if seed_mismatch:assert isinstance(plan.get('DEV_seed_mismatch_explanation'),str) and plan['DEV_seed_mismatch_explanation'].strip(), 'Explain metadata seed differences before GT; do not claim same seed or infer IDs'
 devscore=ROOT/'research/round50/independent_score/dev336_attempt001';ds=read(devscore/'SEALED.json')
 for rel,h in ds['files'].items():assert sha(devscore/rel)==h
 devscores_by_oldid={r['id_PRIVATE']:r for r in read(devscore/'ALL336_TASK_ROWS_PRIVATE.json')}
 import ijson
 p5=ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json';assert sha(p5)==read(ROOT/'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json')['scoring_index_sha256'];index={}
 with p5.open('rb') as f:
  for x in ijson.items(f,'item'):
   if x['id'] not in byid:continue
   r=byid[x['id']];assert x['id'] not in index and x['platform']==r['platform'] and x['scene']==r['scene'] and Path(x['meta_path']).resolve()==Path(r['metapath']).resolve() and type(x['object_index']) is int and x['object_index']>=0
   index[x['id']]={k:x[k] for k in ('meta_path','object_index','metadata_container_sha256')}
 assert len(index)==5912;containers={};tasks=[];dev_reproduced=0
 try:
  for i,r in enumerate(rows):
   z=load(pred/'samples_PRIVATE'/(r['id']+'.npz'));assert z['arm_names'].tolist()==ARMS and z['queries'].shape==(3,) and z['selected_original_boxes'].shape==(3,6) and np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']])
   ix=index[r['id']];mp=ix['meta_path']
   if mp not in containers:assert sha(mp)==ix['metadata_container_sha256'];containers[mp]=read(mp)
   obj=containers[mp]['ground_info'][ix['object_index']];assert ' '.join(obj['caption'].replace(',',' ,').split())==r['caption'];gt=transform_gt(obj['bbox_3d'],r['platform']);boxes=z['selected_original_boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);values=candidate_ious(boxes,gt)[0]
   if r['id'] in devmap:
    oldid=devmap[r['id']]['id']
    for j,a in enumerate(ARMS):assert abs(float(values[j])-devscores_by_oldid[oldid]['IoU'][a])<1e-12
    dev_reproduced+=1
   tasks.append({'row':f'F{i+1:05d}','id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':'DEV336' if r['id'] in devmap else 'NONDEV5576','queries':dict(zip(ARMS,map(int,z['queries']))),'IoU':dict(zip(ARMS,map(float,values)))})
 except Exception as exc:
  dump(OUT/'PARTIAL_ROWS_PRIVATE.json',tasks);dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':len(tasks),'no_automatic_GT_rerun_or_method_adjustment':True});raise
 assert dev_reproduced==336;dump(OUT/'ALL5912_TASK_ROWS_PRIVATE.json',tasks);summary={};scenes=[];neg=[]
 for part in ('ALL5912','DEV336','NONDEV5576'):
  rr=tasks if part=='ALL5912' else [r for r in tasks if r['part']==part];assert len(rr)=={'ALL5912':5912,'DEV336':336,'NONDEV5576':5576}[part];ss={}
  for pl in ('drone','quad'):
   pr=[r for r in rr if r['platform']==pl];ss[pl]={}
   for arm in ARMS:
    v=np.array([r['IoU'][arm] for r in pr]);ds=v-np.array([r['IoU']['source'] for r in pr]);do=v-np.array([r['IoU']['old_G'] for r in pr]);ss[pl][arm]={'n':len(pr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(ds.mean()*100),'gain_vs_old_G_pp':float(do.mean()*100),'improved_vs_source':int((ds>1e-10).sum()),'worsened_vs_source':int((ds< -1e-10).sum()),'improved_vs_old_G':int((do>1e-10).sum()),'worsened_vs_old_G':int((do< -1e-10).sum()),'query_changes_vs_source':sum(r['queries'][arm]!=r['queries']['source'] for r in pr),'query_changes_vs_old_G':sum(r['queries'][arm]!=r['queries']['old_G'] for r in pr),'scene_clusters':len({r['scene_PRIVATE'] for r in pr})}
   for scene in sorted({r['scene_PRIVATE'] for r in pr}):
    sr=[r for r in pr if r['scene_PRIVATE']==scene];scenes.append({'part':part,'platform':pl,'scene_PRIVATE':scene,'n':len(sr),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in sr])*100) for a in ARMS}})
  ss['balanced_mIoU_percent']={a:float(np.mean([ss[p][a]['mIoU_percent'] for p in ('drone','quad')])) for a in ARMS};ss['micro_mIoU_percent']={a:float(np.mean([r['IoU'][a] for r in rr])*100) for a in ARMS};summary[part]=ss
 for r in tasks:
  for a,b in itertools.product(ARMS,repeat=2):
   d=r['IoU'][a]-r['IoU'][b]
   if a!=b and d< -1e-10:neg.append({'row':r['row'],'part':r['part'],'platform':r['platform'],'arm':a,'reference':b,'delta_pp':float(d*100)})
 dump(OUT/'TASK_SUMMARY.json',summary);dump(OUT/'ALL_SCENE_SUMMARIES_PRIVATE.json',scenes);dump(OUT/'ALL_ORDERED_NEGATIVE_COMPARISONS.json',neg)
 receipt={'state':'PASS_R50_SINGLE_LOCKED_FULL5912_THREE_ARM_TASK_SCORE','n':5912,'arms':3,'task_values':17736,'ordered_negative_comparisons':len(neg),'DEV336_joined_by_exact_rawtriple_caption_platform_scene_not_ID_or_alias':True,'DEV336_metadata_seed_equal_n':seed_equal,'DEV336_metadata_seed_mismatch_n':seed_mismatch,'DEV_seed_mismatch_explanation':plan.get('DEV_seed_mismatch_explanation') if seed_mismatch else None,'DEV336_all_three_IoUs_reproduced':dev_reproduced,'historically_exposed_full_target_not_new_independent_confirmation':True,'source_boxes_not_refined':True,'no_k_newrho_or_TTA_in_full_method':True,'no_new_method_selection_after_full_labels':True,'no_protected_FINAL_specific_artifacts_read':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(OUT/'SCORING_RECEIPT.json',receipt);dump(OUT/'SEALED.json',{'state':receipt['state'],'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}});print(json.dumps({'receipt':receipt,'ALL5912_balanced':summary['ALL5912']['balanced_mIoU_percent'],'NONDEV5576_balanced':summary['NONDEV5576']['balanced_mIoU_percent']}))
if __name__=='__main__':main()
