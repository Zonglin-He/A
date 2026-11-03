"""Frozen cache-only cue factorial. No media, model, GT or optimizer imports."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files if k in {'arm_names','scores','queries','source_boxes','source_scores','components','cue_query_known','hybrid_arm_names','hybrid_components','hybrid_cue_query_known'}}
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 assert p['state']=='LOCKED_R52_CACHE_FACTORIAL_NO_NEW_METHOD_NO_TTA' and p['n']==336
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h,rel
 rows=read(ROOT/p['manifest']);out=ROOT/p['output'];assert not out.exists();out.mkdir(parents=True)
 roots={k:ROOT/p[k] for k in ['new_cache','old_cache']};seals={k:read(v/'SEALED.json') for k,v in roots.items()}
 def cached(k,sid):
  rel='samples_PRIVATE/'+sid+'.npz';f=roots[k]/rel;assert sha(f)==seals[k]['files'][rel];return load(f)
 arms=p['arms'];statuses=[];diagnostics=[];start=time.perf_counter()
 for row in rows:
  sid=row['id'];g=cached('new_cache',sid);o=cached('old_cache',sid);s=g['source_scores'];boxes=g['source_boxes'];assert exact(s,o['source_scores']) and exact(boxes,o['source_boxes'])
  assert g['components'].shape==(1,3,256);oi=o['hybrid_arm_names'].tolist().index('replace_appearance_c05')
  cubes={'new':g['components'][0],'old':o['hybrid_components'][oi]};known={'new':g['cue_query_known'][0],'old':o['hybrid_cue_query_known'][oi]}
  assert all(v.shape==(3,256) and np.isfinite(v).all() for v in cubes.values())
  scores=[];ss={}
  for arm in arms:
   if arm=='source':f=s.astype(float);active=np.zeros(256,bool)
   elif arm=='external_G':f=o['scores'][o['arm_names'].tolist().index('external_G')].copy();active=np.ones(256,bool)
   else:
    family,subset=arm.split('_');c=cubes[family];f=s.astype(float);active=np.zeros(256,bool)
    for i,cue in enumerate('VPR'):
     if cue in subset:f=f+2*c[i];active|=known[family][i]
   scores.append(f);unavailable=bool(not active.any()) if arm not in ['source','external_G'] else False
   ss[arm]={'unknown':unavailable,'fallback':unavailable,'no_parameter_update':True}
  scores=np.stack(scores);queries=scores.argmax(1).astype(np.int64)
  assert exact(scores[arms.index('new_VPR')],g['scores'][g['arm_names'].tolist().index('grouped_replace_all_c05')])
  assert exact(scores[arms.index('new_VP')],g['scores'][g['arm_names'].tolist().index('grouped_no_relation_c05')])
  assert exact(scores[arms.index('old_VPR')],o['scores'][o['arm_names'].tolist().index('replace_appearance_c05')])
  dest=out/'samples_PRIVATE'/(sid+'.npz');dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,arm_names=np.array(arms),scores=scores,queries=queries,source_boxes=boxes,source_scores=s,selected_original_boxes=boxes[queries])
  statuses.append({'id':sid,'arm_statuses':ss})
  d={'id_PRIVATE':sid,'platform':row['platform'],'cue_statistics':{}}
  for family,c in cubes.items():
   for i,cue in enumerate('VPR'):
    d['cue_statistics'][family+'_'+cue]={'known_queries':int(known[family][i].sum()),'nonzero_queries':int(np.count_nonzero(c[i])),'range_weighted':float(np.ptp(2*c[i])),'max_abs_weighted':float(np.max(np.abs(2*c[i])))}
  diagnostics.append(d)
 dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'CUE_STATS_PRIVATE.json',diagnostics)
 dump(out/'COSTS.json',{'seconds':time.perf_counter()-start,'new_model_forwards':0,'optimizer_steps':0,'raw_media_reads':0,'task_GT_reads':0,'cohort_n':len(rows),'arms':len(arms)})
 dump(out/'SEALED.json',{'state':'R52_SEALED_CACHE_CUE_SUBSETS_NO_GT','n':len(rows),'arm_names':arms,'plan_sha256':sha(a.plan),'files':{x.relative_to(out).as_posix():sha(x) for x in sorted(out.rglob('*')) if x.is_file()}})
 print(json.dumps({'state':'SEALED','n':len(rows),'arms':len(arms),'seconds':time.perf_counter()-start}))
if __name__=='__main__':main()
