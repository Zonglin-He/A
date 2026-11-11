"""Independent saved-scalar and postseal support-proxy audit, no new inference."""
from common67 import *
H=ROOT/'research/round67';p=read(H/'experiment/DESIGN_LOCK.json');tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');methods=read(H/'ALL_METHOD_ROWS.json');scenes=read(H/'ALL_SCENE_ROWS.json');process=read(H/'ALL_SAMPLING_PROCESS_ROWS.json');rows=read(ROOT/p['manifest']);labels=read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')
for m in methods:
 a=m['method'];x=np.array([t['IoU'][a] for t in tasks]);base=np.array([t['IoU']['R59'] for t in tasks]);assert abs(m['balanced_mIoU']-x.mean()*100)<1e-11 and m['improved']==int((x-base>1e-10).sum()) and m['worsened']==int((x-base< -1e-10).sum())
 for pl,key in [('drone','Drone_mIoU'),('quad','Quad_mIoU')]:assert abs(m[key]-np.mean([t['IoU'][a] for t in tasks if t['platform']==pl])*100)<1e-11
for s in scenes:
 ss=sorted({t['scene_PRIVATE'] for t in tasks if t['platform']==s['platform']});part=[t for t in tasks if t['platform']==s['platform'] and t['scene_PRIVATE']==ss[int(s['scene_alias'][-2:])-1]];assert len(part)==s['n'] and abs(np.mean([t['IoU'][s['method']] for t in part])*100-s['mIoU'])<1e-11
for i,(row,label) in enumerate(zip(rows,labels)):
 assert row['id']==label['id_PRIVATE'];z=load(ROOT/p['capture_root']/'samples_PRIVATE'/(row['id']+'.npz'));best=np.argmax(label['candidate_IoUs_PRIVATE']);box=z['A_boxes'][best];gg=label['candidate_IoUs_PRIVATE'][best]>.25
 for j,v in enumerate('ABC'):
  xyz=z[v+'_SA2_xyz'][0];support=np.all(np.abs(xyz-box[:3])<=box[3:]/2,axis=1)
  for k,mode in enumerate(['native','random','feature']):
   pr=process[i*9+j*3+k];ix=z[mode+'_'+v+'_SA3_indices'].astype(int);assert pr['policy']==mode and pr['view']==v and pr['row']=='DEV_'+str(i+1).zfill(5);assert pr['SA3_n']==512 and pr['unique_index_n']==len(np.unique(ix));assert pr['unique_coordinates_n']==len(np.unique(xyz[ix],axis=0));assert pr['postseal_good_box_proxy_available']==gg
   if gg:assert pr['postseal_proxy_SA3_unique_support']==len(np.unique(xyz[ix][support[ix]],axis=0)) and pr['postseal_proxy_SA2_unique_support']==len(np.unique(xyz[support],axis=0))
mi={m['method']:m['balanced_mIoU'] for m in methods};d=read(H/'DECISION.json');assert d['criterion_passed']==all(mi['feature_R59']>mi[a]+1e-10 for a in p['adoption_requires_feature_exceed'])
out=H/'independent_derived_audit_attempt001';assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R67_INDEPENDENT_DERIVED_METHOD_SCENE_AND_ALL_SUPPORT_PROXIES','n':336,'method_rows':len(methods),'scene_rows':len(scenes),'process_rows':len(process),'no_runtime_GT_feedback':True,'no_new_task_scores':True});print({'derived_audit':'PASS'})
