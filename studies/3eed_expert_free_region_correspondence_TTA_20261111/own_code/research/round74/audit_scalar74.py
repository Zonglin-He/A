"""Independent saved scalar verification; no experiment producer imports."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round74'
def rd(p):return json.loads(Path(p).read_text())
tasks=rd(H/'score_all2304_attempt001/ALL_TASK_ROWS_PRIVATE.json');labels={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in rd(R/'research/round69/score_all2304_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};arms=list(tasks[0]['IoU']);checks=0
for t in tasks:
 for k in arms:assert t['IoU'][k]==labels[t['id_PRIVATE']][t['queries'][k]];checks+=1
for label,n in [('val1280',1280),('train1024',1024),('pilot64',64)]:
 root=H/('score_'+label+'_attempt001');tt=rd(root/'ALL_TASK_ROWS_PRIVATE.json');assert len(tt)==n;ng=[]
 for x in tt:
  for k,ref in itertools.permutations(arms,2):
   d=x['IoU'][k]-x['IoU'][ref]
   if d< -1e-10:ng.append({'row':x['row'],'platform':x['platform'],'part':x['part'],'split':x['split'],'arm':k,'reference':ref,'delta_pp':d*100})
 assert ng==rd(root/'ALL_ORDERED_NEGATIVE_COMPARISONS.json')
 summary=rd(root/'TASK_SUMMARY.json')['ALL'+str(n)]
 for pl in ['drone','quad']:
  uu=[x for x in tt if x['platform']==pl]
  for k in arms:
   a=np.array([x['IoU'][k] for x in uu]);s=summary[pl][k];assert abs(s['mIoU_percent']-a.mean()*100)<1e-12;assert abs(s['Acc25_percent']-(a>.25).mean()*100)<1e-12;assert abs(s['Acc50_percent']-(a>.5).mean()*100)<1e-12
   for key in ['unknown','fallback','no_parameter_update']:assert s[key+'_n']==sum(x['statuses'][k][key] for x in uu)
   checks+=6
 for k in arms:assert abs(summary['balanced_mIoU_percent'][k]-np.mean([summary[p][k]['mIoU_percent'] for p in ['drone','quad']]))<1e-12
for a in rd(H/'ALL_METHOD_ROWS.json'):
 tt=tasks if a['scope']=='ALL2304' else [x for x in tasks if x['part']==a['scope'] or x['phase']==a['scope']];tt=tt if a['platform']=='balanced' else [x for x in tt if x['platform']==a['platform']];v=np.array([x['IoU'][a['method']] for x in tt]);assert len(tt)==a['n'];assert abs(a['mIoU_percent']-v.mean()*100)<1e-12;assert abs(a['Acc25_percent']-(v>.25).mean()*100)<1e-12;checks+=3
for a in rd(H/'ALL_SCENE_ROWS.json'):
 allsc=sorted({(x['part'],x['platform'],x['scene_PRIVATE']) for x in tasks});key=allsc[int(a['recording_alias'].split('_')[-1])-1];tt=[x for x in tasks if (x['part'],x['platform'],x['scene_PRIVATE'])==key];assert len(tt)==a['n'];assert abs(a['mIoU_percent']-np.mean([x['IoU'][a['method']] for x in tt])*100)<1e-12;checks+=2
media=rd(R/'research/round69/input/SELECTED_MEDIA_HASHES_PRIVATE.json');frames={}
for x in tasks:
 m=media[x['id_PRIVATE']];key=(x['part'],x['platform'],m['image_sha256'],m['point_sha256']);frames.setdefault(key,[]).append(x)
ff=[]
for i,(f,tt) in enumerate(sorted(frames.items())):
 for k in arms:ff.append({'frame_alias':f'FRAME_{i+1:05d}','part':f[0],'platform':f[1],'descriptions':len(tt),'method':k,'mean_IoU':float(np.mean([x['IoU'][k] for x in tt]))})
assert ff==rd(H/'ALL_FRAME_CLUSTER_ROWS.json')
steps=rd(H/'ALL_OPTIMIZER_STEP_ROWS.json');co=rd(H/'ALL_COVERAGE_ROWS.json');d=rd(H/'COHORT_DIAGNOSTICS.json');assert len(co)==2304 and len(steps)==6912 and d['accepted_updates']==sum(s['accepted'] for s in steps)==6909;assert d['RGB_noRGB_task_differences']==sum(t['queries']['corr_RGB']!=t['queries']['corr_noRGB'] for t in tasks);assert not any(s['accepted'] and s['loss_after']>s['loss_before'] for s in steps)
out=H/'independent_scalar_audit_attempt001';assert not out.exists();out.mkdir();v={'state':'PASS_R74_ALL2304_SAVED_TASK_NEGATIVE_METHOD_SCENE_FRAME_AND_COVERAGE_SCALARS','n':2304,'task_values':len(tasks)*len(arms),'checks':checks,'frame_clusters':len(frames),'recordings':len(allsc),'accepted_updates':6909,'no_new_GT_geometry_or_model_operations':True};(out/'AUDIT_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');print(v)
