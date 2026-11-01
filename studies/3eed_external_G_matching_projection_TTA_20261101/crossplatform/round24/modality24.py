"""Matched-crop Qwen removal and CLIP replacement, no task labels."""
import sys,time,copy,numpy as np
from common24 import *
sys.path.insert(0,str(ROOT/'crossplatform/round8'))
from semantic_encoder import FrozenSemanticEncoder
from observations import load
from visualZ import crops
from reference21_v3 import crop as reference_crop
from rank23 import relation_value
from detect import rect_iou

OBS_CACHE={}
def observe(r):
 if r['id'] not in OBS_CACHE:
  OBS_CACHE.clear();OBS_CACHE[r['id']]=load(r)
 return OBS_CACHE[r['id']]
def paths(r):
 if r['stage']=='M':b=BASE;suf='M64'
 else:b=ROOT/'crossplatform/round23';suf='272'
 return {key:b/(name+suf+'_attempt001') for key,name in [('prep','prepared'),('text','text'),('dino','dino'),('evidence','evidence')]}
def components(r,ev,kind='clip',scale=20.):
 p=paths(r);prep=read(p['prep']/'cases_PRIVATE'/(r['id']+'.json'));g=read(p['text']/'cases_PRIVATE'/(r['id']+'.json'));z=feature(r);a=load_npz(p['prep']/'arrays_PRIVATE'/(r['id']+'.npz'));pool=prep['mask_pool_PRIVATE'];vs=np.zeros(256);rs=np.zeros(256);ps=np.zeros(256)
 if not pool:return vs,ps,rs
 frame,scene,binding=observe(r);W,H=scene['image_size_wh'];q0=int(z['scores'].argmax());v=[];rv=[]
 if kind=='clip':
  ee={'references':{}}
  for key,refs in ev['references'].items():ee['references'][key]=[dict(x,full={'log_odds_yes_no':x['delta']*scale},visual_evidence=x['delta']*scale) for x in refs]
 elif kind=='no_qwen':
  # Retain independently detected reference boxes and geometry, remove all Qwen
  # appearance/identity confidence. Confidence is detector-only, not fabricated Yes.
  d=read(p['dino']/'cases_PRIVATE'/(r['id']+'.json'));ee={'references':{}}
  for ref in g['references']:
   det=d['objects'].get(ref['entity']);ee['references'][ref['entity']]=[] if not det else [dict(box_PRIVATE=det['original']['boxes_PRIVATE'][j],detector_score=float(det['original']['scores'][j]),full={'log_odds_yes_no':30.},visual_evidence=30.) for j in det['stable_indices'][:3]]
 else:raise ValueError(kind)
 for m in pool:
  v.append(float(np.mean([a['delta']*scale for a in ev['targets_PRIVATE'][str(m)]])) if kind=='clip' else 0.)
  rv.append(relation_value(g,scene['objects'][m],ee,W,H))
 weights=a['weights'];mapped=weights.sum(1)>0;v=np.array(v);rv=np.array(rv);vv=weights@v;rr=weights@rv;anchor_v=vv[q0] if mapped[q0] else float(np.median(v));anchor_r=rr[q0] if mapped[q0] else np.median(rv,axis=0);vs[mapped]=np.clip((vv[mapped]-anchor_v)/2,-2,2);ps[mapped]=np.clip(rr[mapped,0]-anchor_r[0],-2,2);rs[mapped]=np.clip(rr[mapped,1]-anchor_r[1],-2,2)
 return vs,ps,rs

def main():
 import torch
 from PIL import Image
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;out=BASE/'clip336_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();(out/'banks_PRIVATE').mkdir();start=time.perf_counter();model=FrozenSemanticEncoder(device='cuda');count=0;texts=0;truncated=0;receipts=[]
 for i,r in enumerate(rows()):
  p=paths(r);prep=read(p['prep']/'cases_PRIVATE'/(r['id']+'.json'));g=read(p['text']/'cases_PRIVATE'/(r['id']+'.json'));d=read(p['dino']/'cases_PRIVATE'/(r['id']+'.json'));outcase={'alias':r['alias'],'targets_PRIVATE':{},'references':{}};ims=[];queries=[];tags=[]
  if prep['mask_pool_PRIVATE']:
   frame,scene,binding=observe(r);assert binding==prep['bindings']
   for m in prep['mask_pool_PRIVATE']:
    for k,image in enumerate(crops(frame['rgb'],scene['objects'][m])):ims += [image,Image.new('RGB',image.size,(127,127,127))];queries += [g['subject_query']]*2;tags.append(('target',str(m),k,None))
   for ref in g['references']:
    det=d['objects'][ref['entity']]
    for j in det['stable_indices'][:3]:
     box=det['original']['boxes_PRIVATE'][j];image=reference_crop(frame['rgb'],box);ims += [image,Image.new('RGB',image.size,(127,127,127))];queries += [ref['phrase']]*2;tags.append(('ref',ref['entity'],j,dict(box_PRIVATE=box,detector_score=float(det['original']['scores'][j]),proposal_index=j)))
   unique=list(dict.fromkeys(queries));te,tr=model.encode_texts(unique);im=model.encode_crops(ims);lookup={q:j for j,q in enumerate(unique)};sim=(im*te[[lookup[q] for q in queries]]).sum(-1).numpy();truncated+=tr['truncated_count'];count+=len(ims);texts+=len(unique);receipts.append(dict(alias=r['alias'],**tr))
   for k,(typ,key,j,extra) in enumerate(tags):
    rec={'view_or_proposal':j,'cosine':float(sim[2*k]),'gray_cosine':float(sim[2*k+1]),'delta':float(sim[2*k]-sim[2*k+1]),'crop_sha256':arrsha(np.asarray(ims[2*k]))}
    if extra:rec.update(extra)
    outcase['targets_PRIVATE' if typ=='target' else 'references'].setdefault(key,[]).append(rec)
  dump(out/'cases_PRIVATE'/(r['id']+'.json'),outcase);v,p0,rel=components(r,outcase,'no_qwen');payload={'no_qwen_vision':v,'no_qwen_position':p0,'no_qwen_relation':rel}
  for scale in [5,10,20,40]:
   v,p0,rel=components(r,outcase,'clip',scale);payload.update({f'clip{scale}_vision':v,f'clip{scale}_position':p0,f'clip{scale}_relation':rel})
  np.savez_compressed(out/'banks_PRIVATE'/(r['id']+'.npz'),**payload)
  if (i+1)%32==0:print('CLIP24',i+1,count,'seconds',round(time.perf_counter()-start,1),flush=True)
 model._assert_frozen();dump(out/'TOKEN_RECEIPTS_PRIVATE.json',receipts);seal(out,'R24_matched_candidates_crops_texts_noQwen_and_frozenCLIP_controls',n=336,image_encodings=count,text_encodings=texts,truncated_text_queries=truncated,model_unchanged=True,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),resource=model.resource);print('CLIP24_SEALED',flush=True)
if __name__=='__main__':main()
