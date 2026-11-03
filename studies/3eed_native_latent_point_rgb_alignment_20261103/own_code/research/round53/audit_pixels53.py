"""Four hash-selected RGB/centroid checks using the original color function.

The original descriptor's final mean is replaced with a weighted mean through
an isolated numpy proxy; HSV/Lab/prototype code is neither copied nor changed.
Prior-audited R51 spatial operator is an unchanged dependency, not a new claim.
"""
import argparse,copy,hashlib,json,sys,time,types
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round51'));sys.path.insert(0,str(ROOT/'research/round51'))
from attributes51 import color_support,COLOR_RGB_PROTOTYPES
from evidence51_fidelity import object_geometry,raw_relations
from audit51 import measured
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class MeanProxy:
 def __init__(self,w):self.w=w
 def __getattr__(self,name):return getattr(np,name)
 def mean(self,a):return np.dot(self.w,a)/self.w.sum()
def measure(pixels,w,scene,ns,index):
 total=w.sum();n=total*total/(w@w) if total>0 else 0.;known=n>=8-1e-10 and ns>=8
 f=types.FunctionType(color_support.__code__,{**color_support.__globals__,'np':MeanProxy(w)})
 obs=f(pixels) if total>0 else {k:None for k in COLOR_RGB_PROTOTYPES};value={k:None for k in COLOR_RGB_PROTOTYPES}
 if known:
  for k in value:
   a=(n*obs[k]+.5)/(n+1);b=(ns*scene[k]+.5)/(ns+1);value[k]=float(np.clip(np.log(a/(1-a))-np.log(b/(1-b)),-4,4))
 return {'color_logodds_relative_scene_visible':value,'color_evidence_reliability':n/(n+8) if known else 0.}
def evidence(items,requests,entity):
 words=next((x['recognized_color_words'] for x in requests if x['entity']==entity),[]);values=[];valid=[]
 for r in items:
  have=[r['color_logodds_relative_scene_visible'].get(w) for w in words];have=[x for x in have if x is not None];values.append(np.mean(have)*r['color_evidence_reliability'] if have else 0.);valid.append(bool(have))
 return np.asarray(values),np.asarray(valid,bool)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 rows=read(ROOT/p['manifest']);selected=[]
 for platform in ['drone','quad']:selected+=sorted([r for r in rows if r['platform']==platform],key=lambda r:hashlib.sha256(('R53_pixel_measurement_audit_v1:'+r['id']).encode()).digest())[:2]
 out=ROOT/'research/round53/pixel_measurement_audit_attempt001';assert not out.exists();out.mkdir();start=time.perf_counter();errors=[];groups=0
 for row in selected:
  sid=row['id'];groot=ROOT/p['grouped_root'];proot=ROOT/p['prepared_root'];pred=ROOT/p['prediction_root'];attrs=read(groot/'attributes_PRIVATE'/(sid+'.json'));refs=read(groot/'references_PRIVATE'/(sid+'.json'))['references_by_factor'][0];pc=read(proot/'cases_PRIVATE'/(sid+'.json'));assert sha(row['imagepath'])==pc['raw_bindings']['image_sha256']
  with Image.open(row['imagepath']) as im:rgb=np.asarray(im.convert('RGB'))
  with np.load(groot/'carriers_PRIVATE'/(sid+'.npz'),allow_pickle=False) as z:carrier={k:z[k].copy() for k in z.files}
  with np.load(groot/'samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as z:g={k:z[k].copy() for k in ['source_scores','query_instance_conditional_weights']}
  with np.load(proot/'arrays_PRIVATE'/(sid+'.npz'),allow_pickle=False) as z:vis=z['visible_raw_point_indices'];uv=z['visible_uv'];index={int(x):j for j,x in enumerate(vis)}
  with np.load(pred/'samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as target,np.load(pred/'supports_PRIVATE'/(sid+'.npz'),allow_pickle=False) as support:
   M=len(carrier['group_medoid_queries']);W,H=pc['image_size_wh'];ent=['T']+list(refs);scene=attrs['scene_visible_color_soft_support'];ns=attrs['n_scene_visible_pixels']
   for vi,variant in enumerate(['pre','native','shuffle']):
    items={e:[] for e in ent};positions={e:[] for e in ent};objects=object_geometry(carrier,(W,H))
    for m in range(M):
     lo,hi=carrier['instance_crop_mask_pixel_offsets'][m:m+2];px=carrier['instance_crop_mask_pixel_xy'][lo:hi];lo,hi=carrier['instance_visible_point_offsets'][m:m+2];ids=carrier['instance_visible_point_indices'][lo:hi];uu=uv[[index[int(x)] for x in ids]];pixels=rgb[px[:,1],px[:,0]]
     for e in ent:
      w=support[variant+'__'+e+'__'+str(m)];items[e].append(measure(pixels,w,scene,ns,m));positions[e].append(np.average(uu,axis=0,weights=w) if len(w) and w.sum()>0 else None)
     wt=support[variant+'__T__'+str(m)]
     if len(wt) and wt.sum()>0:objects[m]['xy']=(np.average(px.astype(float),axis=0,weights=wt)+.5)/[W,H]
     groups+=1
    hypotheses={}
    for e,info in refs.items():
     dist=info['distribution'];keep=dist['source_top3_instance_ids'];prob=np.asarray(dist['source_probabilities']);ve,valid=evidence(items[e],attrs['entity_attribute_requests'],e);updated=np.asarray([prob[j]*np.exp(np.clip(.5*ve[j],-8,8)) if valid[j] else prob[j] for j in keep]);den=updated.sum()+dist['source_unknown_probability']+dist['source_truncated_probability'];updated/=den;by={h['instance_index']:h for h in info['hypotheses']};hh=[]
     for j,v in zip(keep,updated):
      h=copy.deepcopy(by[j]);h['probability']=float(v);h['xy_norm2']=(positions[e][j]/[W,H]).tolist() if positions[e][j] is not None else None;hh.append(h)
     hypotheses[e]=hh
    e,known=evidence(items['T'],attrs['entity_attribute_requests'],'T');pr,pk=raw_relations(attrs['text_graph_unchanged'],objects,hypotheses,(W,H));_,cue,_,_,_=measured(g['source_scores'],g['query_instance_conditional_weights'],.5*e,pr,known,pk);error=float(np.max(np.abs(cue-target['components'][vi])));assert error<1e-10,error;errors.append(error)
 result={'state':'PASS_R53_ORIGINAL_DESCRIPTOR_WEIGHTED_RGB_AND_CENTROID_AUDIT','hash_selected_observations':len(selected),'rows_per_platform':2,'variant_case_comparisons':len(errors),'group_variant_measurements':groups,'max_cue_absolute_error':max(errors),'original_RGB_hashes_checked':True,'original_color_function_reused_with_isolated_weighted_mean':True,'fixed_R51_spatial_dependency_not_reimplemented':True,'new_models_or_GT':0,'seconds':time.perf_counter()-start,'prediction_seal_sha256':sha(ROOT/p['prediction_root']/'SEALED.json'),'program_sha256':sha(__file__),'plan_sha256':sha(a.plan)};(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
