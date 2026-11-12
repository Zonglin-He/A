"""Saved-only independent task/negative aggregation and weight/update diagnostics."""
from core79 import *
import itertools
P=B79/'prediction464_attempt001';S=H79/'score_dev336_attempt001';ss=read(S/'SEALED.json')
for f,h in ss['files'].items():assert sha(S/f)==h
tt=read(S/'ALL_TASK_ROWS_PRIVATE.json');su=read(S/'TASK_SUMMARY.json');sc=read(S/'ALL_SCENE_SUMMARIES_PRIVATE.json');neg=read(S/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');method=[];scenes=[];checks=0
for scope,rows in [('ALL336',tt)]+[(k,[r for r in tt if r['split']==k]) for k in sorted({r['split'] for r in tt}) if k!='ALL336']:
 for pl in ['drone','quad']:
  rr=[r for r in rows if r['platform']==pl]
  for a in ARMS79:
   v=np.array([r['IoU'][a] for r in rr]);z=su[scope][pl][a];assert abs(v.mean()*100-z['mIoU_percent'])<1e-12 and np.mean(v>.25)*100==z['Acc25_percent'] and np.mean(v>.5)*100==z['Acc50_percent'];checks+=3;method.append({'scope':scope,'platform':pl,'arm':a,**z})
expected=[(r['row'],a,b,100*(r['IoU'][a]-r['IoU'][b])) for r in tt for a,b in itertools.permutations(ARMS79,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];assert expected==[(r['row'],r['arm'],r['reference'],r['delta_pp']) for r in neg]
names={pl:sorted({r['scene_PRIVATE'] for r in tt if r['platform']==pl}) for pl in ['drone','quad']}
for r in sc:
 for a in ARMS79:scenes.append({'scope':r['scope'],'platform':r['platform'],'scene_alias':('D' if r['platform']=='drone' else 'Q')+'_S'+str(names[r['platform']].index(r['scene_PRIVATE'])+1).zfill(2),'n':r['n'],'arm':a,'mIoU_percent':r['mIoU_percent'][a]})
source=read(H79/'SOURCE128_TASK_SCALARS.json');srcsum=read(H79/'SOURCE128_SUMMARY.json');srcarms=list(srcsum)
for a in srcarms:
 v=np.array([r['IoU'] for r in source if r['arm']==a]);assert len(v)==128 and abs(v.mean()*100-srcsum[a]['mIoU_percent'])<1e-12
weightrows=[]
for i,r in enumerate(source_rows('VAL128')+target_rows()):
 rec=read(P/'records_PRIVATE'/(r['id']+'.json'));wp=load(P/'weights_PRIVATE'/(r['id']+'.npz'));case=('S'+str(i+1).zfill(5)) if i<128 else 'R'+str(i-127).zfill(5);rP=wp['point_saliency'];rI=wp['image_saliency'];row={'row':case,'platform':r['platform'],'pair_count':rec['saliency']['pair_count'],'point_saliency_sum':float(rP.sum()),'image_saliency_sum':float(rI.sum()),'uniform_fallback_weight_sets':rec['fallback_weight_sets']}
 for kind in ['query','shuffle']:
  for typ in ['P','I','D']:
   peak=[];neff=[]
   for j in [0,1]:
    w=wp[kind+'_'+str(j)+'_'+typ];n=int((w>0).sum());peak.append(float(w.max()*n) if n else 0.);neff.append(float(1/max(np.square(w.astype(float)).sum(),1e-300))/n if n else 0.)
   row[kind+'_'+typ+'_peak_times_uniform']=float(np.mean(peak));row[kind+'_'+typ+'_effective_fraction']=float(np.mean(neff))
 weightrows.append(row)
updates=read(H79/'ALL_ENDPOINT_SCALAR_ROWS.json');us=[]
for pl in ['waymo','drone','quad']:
 for a in ['uniform_sgd1','query_raw_sgd1','query_radius_sgd1','shuffle_radius_sgd1','wrong_binding_radius_sgd1']:
  rr=[r for r in updates if r['platform']==pl and r['arm']==a];us.append({'platform':pl,'arm':a,'n':len(rr),'parameter_L2_mean':float(np.mean([r['parameter_step_L2'] for r in rr])),'parameter_L2_median':float(np.median([r['parameter_step_L2'] for r in rr])),'query_changes':sum(r['query_changed'] for r in rr),'mean_cosine_to_uniform':float(np.mean([r['cosine_to_uniform_proposal'] for r in rr if r['cosine_to_uniform_proposal'] is not None])),'mean_auxiliary_after':float(np.mean([r['auxiliary_after'] for r in rr]))})
for n,x in [('ALL_METHOD_ROWS',method),('ALL_SCENE_ROWS',scenes),('WEIGHT_DIAGNOSTIC_ROWS',weightrows),('UPDATE_SUMMARY_ROWS',us)]:dump(H79/(n+'.json'),x)
out=H79/'independent_scalar_audit_attempt001';out.mkdir();res={'state':'PASS_R79_ALL336_SAVED_TASK_SOURCE_AND_ORDERED_NEGATIVE_SCALARS','n':336,'task_values':4032,'ordered_negatives':len(neg),'source_task_values':1280,'source_ordered_negatives':len(read(H79/'SOURCE128_ORDERED_NEGATIVES.json')),'checks':checks,'no_new_GT_geometry_or_models':True};dump(out/'AUDIT_RESULT.json',res);print(res)
