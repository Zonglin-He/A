"""Real raw observations, no external vision model/cache; parallel CPU workers."""
from common72 import *
from geometry72 import scene,supports
from text72 import parse
from constraints72 import build
from scipy.special import softmax,logsumexp
from concurrent.futures import ProcessPoolExecutor
import argparse,time,os

def worker(r):
 z=prefix(r,('boxes','scores','tokens','query','token_mask'));a=prep(r);text=parse(r['caption'],len(z['tokens']));s=scene(r);a['image_width']=s['wh'][0];tm=supports(s,z['boxes'],a,text['target']);rm=supports(s,z['boxes'],a,text['reference']);qref=None
 if text['ref_mask'].any() and (~text['ref_mask']).any():
  l=np.einsum('ik,jk->ij',z['query'].astype(float),z['tokens'].astype(float))/.07;m=text['ref_mask'];qref=softmax(logsumexp(l[:,m],1)-logsumexp(l[:,~m],1))
 N,rel=build(a,z['boxes'],tm,rm,text,qref,r['id']);data={**N,**{'target_'+k:v for k,v in tm.items()},**{'reference_'+k:v for k,v in rm.items()},'query_group':a['query_group'],'group_representatives':a['group_representatives'],'rects':a['rects'],'project_reliability':a['project_reliability'],'ref_mask':text['ref_mask'],'reference_prob':np.zeros(256) if qref is None else qref,'image_width':s['wh'][0]};group_count=len(a['rects']);source_pg=np.bincount(a['query_group'],weights=z['scores'],minlength=group_count);rec={'id_PRIVATE':r['id'],'target_request_PRIVATE':text['target'],'reference_request_PRIVATE':text['reference'],'reference_record':rel,'target_entity_known':text['target_entity_known'],'target_color_parsed':text['target']['color'] is not None,'reference_color_parsed':text['reference']['color'] is not None,'relation_literal':text['has_relation_literal'],'explicit_image_relation':text['explicit_image_frame'],'visible_graph_points':len(s['xyz']),'strict_components':int(s['labels'][0].max()+1) if len(s['xyz']) else 0,'loose_components':int(s['labels'][1].max()+1) if len(s['xyz']) else 0,'supported_groups':int(tm['known'].sum()),'box_supported_groups':int(tm['box_known'].sum()),'main_rejected_groups':int(N['main_N'].sum()),'source_rejected_mass':float(source_pg[N['main_N']].sum()),'source_top1_rejected':bool(N['main_N'][a['query_group'][z['scores'].argmax()]]),'image_sha256':sha(r['imagepath']),'point_sha256':sha(r['pointpath']),'no_CLIP_SAM_DINO_VLM_or_GT_used':True};return r['id'],data,s,rec

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();p=read(H/'experiment/DESIGN_LOCK.json');pin(p);rr=rows();co=cohort_map()
 if args.engineering:
  import re
  eligible=[r for r in rr if re.search(r'\b(red|green|blue|yellow|shirt|jacket|coat|pants)\b',r['caption'],re.I)];rr=[min([r for r in eligible if co[r['id']]==c and r['platform']==pl],key=lambda r:hashlib.sha256(('R72eng|'+r['id']).encode()).hexdigest()) for c in ['VAL1280','TRAIN1024'] for pl in ['drone','quad']]
 out=H/'engineering_evidence4_attempt001' if args.engineering else ROOT/p['evidence_root'];assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','graphs_PRIVATE','cases_PRIVATE']:(out/f).mkdir()
 start=time.perf_counter();active=support=0
 with ProcessPoolExecutor(max_workers=4) as ex:
  for i,(sid,data,s,rec) in enumerate(ex.map(worker,rr,chunksize=1)):
   np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),**data);np.savez_compressed(out/'graphs_PRIVATE'/(sid+'.npz'),**s);dump(out/'cases_PRIVATE'/(sid+'.json'),rec);active+=rec['main_rejected_groups']>0;support+=rec['supported_groups']>0
   if(i+1)%64==0:print('EVID72',i+1,len(rr),active,support,round(time.perf_counter()-start,1),flush=True)
 dump(out/'COUNTS.json',{'n':len(rr),'active_rejection_inputs':active,'supported_inputs':support,'new_external_models':0,'source_full_forwards':0,'GT_labels_used':False,'seconds':time.perf_counter()-start});seal(out,'R72_GEOMETRIC_SURFACE_LANGUAGE_EXCLUSIONS_BEFORE_ANY_TASK_LOOKUP',n=len(rr));print(read(out/'COUNTS.json'))
