"""Independent conflict direction/color/measurement audit, no task labels."""
from pathlib import Path
import json,hashlib,sys,time,numpy as np
R=Path(__file__).resolve().parents[2]
def rd(p):return json.loads(Path(p).read_text())
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
rows=rd(R/'research/round24/input/ALL336_PRIVATE.json');out=R/'research/round55/observation_audit_attempt001';assert not out.exists();out.mkdir();ev=R/'crossplatform/round55/conflicts336_attempt001';es=rd(ev/'SEALED.json');start=time.perf_counter();color_checks=0;spatial_checks=0;raw_checks=0;maxerr=0.
colors=['red','orange','yellow','green','cyan','blue','purple','pink','brown','beige','white','gray','black'];hues={'red':0.,'orange':.08,'yellow':.16,'green':.33,'cyan':.5,'blue':.66,'purple':.78,'pink':.91}
# No outcome-based case selection: two/platform by fixed input hash.
selected=set()
for pl in ['drone','quad']:selected.update(r['id'] for r in sorted([r for r in rows if r['platform']==pl],key=lambda x:hashlib.sha256(('R55_RAW_AUDIT:'+x['id']).encode()).digest())[:2])
for row in rows:
 sid=row['id'];p=ev/'samples_PRIVATE'/(sid+'.npz');assert h(p)==es['files']['samples_PRIVATE/'+sid+'.npz'];a=load(p);rec=rd(ev/'cases_PRIVATE'/(sid+'.json'));requests=rec['requests_PRIVATE'];names=rec['entity_names_PRIVATE']
 for key,expected in [('color_means','color_conflict'),('shuffled_color_means','shuffled_color_conflict')]:
  m=a[key];value=np.zeros((len(requests),256));known=(a['counts']>=16).all(0)&(a['cover']>=.35).all(0)
  for entity,words in enumerate(requests):
   if len(words)!=1 or words[0] not in colors:continue
   word=words[0];opposite=[]
   for k,c in enumerate(colors):
    if word in hues and c in hues and min(abs(hues[c]-hues[word]),1-abs(hues[c]-hues[word]))>=.24:opposite.append(k)
    if (word,c) in [('white','black'),('black','white')]:opposite.append(k)
   if not opposite:continue
   alt=m[:,:,opposite].max(-1);actual=m[:,:,colors.index(word)];test=known&(alt>=.45).all(0)&(actual<=.20).all(0)&((alt-actual)>=.30).all(0);w=np.minimum(.95,a['counts'].min(0)/(a['counts'].min(0)+16))*np.minimum(np.maximum((alt-actual).min(0)/.5,0),1);value[entity,test]=w[test]
  assert np.array_equal(value,a[expected]);color_checks+=value.size
 lo=a['intervals'][:,0];hi=a['intervals'][:,1];known=a['geometry_valid'];eligible=(a['support_overlap']<=.1)&known[:,None]&known[None,:];g=np.zeros_like(a['geometry_conflict'])
 for edge in rec['edges_PRIVATE']:
  if edge['frame']!='camera_image':continue
  s,t=edge['subject'],edge['reference'];pred=edge['predicate']
  if pred not in ['LEFT_OF','RIGHT_OF']:continue
  if s=='T' and t in names[1:]:r=t;direction=1
  elif t=='T' and s in names[1:]:r=s;direction=-1
  else:continue
  if pred=='RIGHT_OF':direction=-direction
  bad=(lo[:,None]-hi[None,:]>.02) if direction==1 else (lo[None,:]-hi[:,None]>.02)
  index=names.index(r)-1;g[index]=np.maximum(g[index],np.where(eligible&bad,.9,0))
 assert np.array_equal(g,a['geometry_conflict']);spatial_checks+=g.size
 assert all(not np.diag(x).any() for x in g)
 if sid in selected:
  sys.path.insert(0,str(R/'research/round53'));from support53 import pixel_descriptors
  visible=load(R/'crossplatform/round51/prepared336_attempt001/arrays_PRIVATE'/(sid+'.npz'))['visible_raw_point_indices'];p=Path(row['pointpath']);raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4);xyz=raw[visible,:3].astype(float)
  if row['platform']=='drone':xyz[:,2]+=1.8
  assert np.array_equal(xyz,a['visible_xyz'])
  lat=load(R/'crossplatform/round53/latent336_attempt001/samples_PRIVATE'/(sid+'.npz'));boxes=lat['boxes'].astype(float);desc=pixel_descriptors(a['visible_rgb']);sh=desc[a['visible_RGB_permutation']];counts=np.zeros((3,256),int);means=np.zeros_like(a['color_means']);shmeans=means.copy()
  # Invalid/truncated projected boxes cannot provide evidence; production counts then zero.
  rr=a['projection_rects'];W,H=rd(R/'crossplatform/round51/prepared336_attempt001/cases_PRIVATE'/(sid+'.json'))['image_size_wh'];valid=(rr[:,0]>=0)&(rr[:,1]>=0)&(rr[:,2]<W)&(rr[:,3]<H)&(boxes[:,3:]>0).all(1)
  for qi,box in enumerate(boxes):
   for vi,scale in enumerate([.85,1.,1.15]):
    ids=np.flatnonzero((np.abs(xyz-box[:3])<=box[3:]*scale/2).all(1)) if valid[qi] and a['counts'][:,qi].any() else np.empty(0,int);counts[vi,qi]=len(ids)
    if len(ids):means[vi,qi]=desc[ids].mean(0);shmeans[vi,qi]=sh[ids].mean(0)
  assert np.array_equal(counts,a['counts']);err=max(float(abs(means-a['color_means']).max()),float(abs(shmeans-a['shuffled_color_means']).max()));maxerr=max(maxerr,err);assert err<1e-12;raw_checks+=3*256
result={'state':'PASS_R55_INDEPENDENT_DIRECTED_RELATION_COLOR_AND_RAW_SUPPORT_AUDIT','n':336,'color_entity_query_checks':color_checks,'directed_relation_pair_checks':spatial_checks,'hash_selected_raw_cases':4,'raw_box_scale_checks':raw_checks,'max_raw_descriptor_error':maxerr,'camera_and_prototype_functions_common_dependency':True,'no_instance_identity_or_true_reference_GT_certification':True,'GT_reads':0,'models':0,'seconds':time.perf_counter()-start,'program_sha256':h(__file__)}
(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
