"""Independent saved scalar totals, all negatives, scenes, static/TTA separation."""
from core83 import *
import itertools,collections
methods=[];scenes=[];inter=[];tails=[];nall=nn=0
for part,n in [('VAL1280',1280),('TRAIN1024',1024)]:
 F=H83/('score_'+part.lower()+'_attempt001');ss=read(F/'SEALED.json')
 for f,h in ss['files'].items():assert sha(F/f)==h
 rr=read(F/'ALL_TASK_ROWS_PRIVATE.json');su=read(F/'TASK_SUMMARY.json')['ALL'+str(n)];ng=read(F/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');assert [(r['row'],r['arm'],r['reference'],r['delta_pp']) for r in ng]==[(r['row'],a,b,100*(r['IoU'][a]-r['IoU'][b])) for r in rr for a,b in itertools.permutations(ARMS83,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];nall+=n;nn+=len(ng);counts=collections.Counter(r['platform'] for r in rr)
 for pl in ['drone','quad']:
  rows=[r for r in rr if r['platform']==pl]
  for a in ARMS83:
   v=np.array([r['IoU'][a] for r in rows]);d=su[pl][a];assert abs(v.mean()*100-d['mIoU_percent'])<1e-12 and (v>.25).mean()*100==d['Acc25_percent'] and (v>.5).mean()*100==d['Acc50_percent'];methods.append({'cohort':part,'platform':pl,'arm':a,**d})
  for j,scene in enumerate(sorted({r['scene_PRIVATE'] for r in rows})):
   sub=[r for r in rows if r['scene_PRIVATE']==scene]
   for a in ARMS83:scenes.append({'cohort':part,'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(j+1).zfill(2),'n':len(sub),'arm':a,'mIoU_percent':float(np.mean([r['IoU'][a] for r in sub])*100)})
 for pl in ['drone','quad','balanced']:
  d=su['balanced_mIoU_percent'] if pl=='balanced' else {a:su[pl][a]['mIoU_percent'] for a in ARMS83};native=d['BT_native']-d['frozen_native']
  for readout in ['native','background','role','mismatch']:
   f='frozen_'+readout;t='BT_'+readout;inter.append({'cohort':part,'platform':pl,'readout':readout,'frozen_mIoU_percent':d[f],'BT_mIoU_percent':d[t],'TTA_gain_pp':d[t]-d[f],'interaction_vs_native_pp':d[t]-d[f]-native,'frozen_readout_gain_vs_native_pp':d[f]-d['frozen_native'],'BT_readout_gain_vs_native_pp':d[t]-d['BT_native'],'frozen_attribute_gain_vs_background_pp':d[f]-d['frozen_background'],'BT_attribute_gain_vs_background_pp':d[t]-d['BT_background'],'frozen_gain_vs_mismatched_roles_pp':d[f]-d['frozen_mismatch'],'BT_gain_vs_mismatched_roles_pp':d[t]-d['BT_mismatch']})
 for a,ref in [('frozen_background','frozen_native'),('BT_background','BT_native'),('frozen_role','frozen_native'),('BT_role','BT_native'),('frozen_role','frozen_background'),('BT_role','BT_background'),('frozen_role','frozen_mismatch'),('BT_role','BT_mismatch')]:
  vals=np.array([50*(r['IoU'][a]-r['IoU'][ref])/counts[r['platform']] for r in rr]);ds=np.array([r['IoU'][a]-r['IoU'][ref] for r in rr]);tails.append({'cohort':part,'arm':a,'reference':ref,'gain_pp':float(vals.sum()),'improved':int((ds>1e-10).sum()),'worsened':int((ds< -1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries'][ref] for r in rr),'largest_positive_contribution_pp':float(vals.max()),'excluding_largest_positive_original_denominator_pp':float(vals.sum()-max(0,vals.max()))})
dump(H83/'ALL_METHOD_ROWS.json',methods);dump(H83/'ALL_SCENE_ROWS.json',scenes);dump(H83/'READOUT_INTERACTION_ROWS.json',inter);dump(H83/'TASK_EFFECT_AND_TAIL_ROWS.json',tails);O=H83/'independent_scalar_audit_attempt001';O.mkdir();result={'state':'PASS_R83_ALL_SCALAR_VALUES_SCENES_NEGATIVES_AND_FACTORIAL_INTERACTIONS','n':nall,'task_values':nall*len(ARMS83),'ordered_negatives':nn,'no_new_GT_models_media_or_geometry':True};dump(O/'AUDIT_RESULT.json',result);print(result);print([r for r in inter if r['platform']=='balanced'])
