"""Bounded original-C1 process readback and identity evidence preparation."""
import sys,json,time,hashlib,argparse,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
PARENT=ROOT/'artifacts/c1_concept_local_v1'
OUT=ROOT/'artifacts/c1_correction_mechanism_v1'
QUERY='spatial.query_residual'
def stem(r):return r['key'].replace(':','_')
def digest(k):return hashlib.sha256(k.encode()).hexdigest()
def original(r):return load(r['parent_file'])
def prepare():
 import torch
 from scripts.c1_concept_state_diagnostic_v1 import check_parent
 from scripts.analyze_spatial_reference_absorption_v1 import score
 from vg_tta.box_stability_diagnostics_v1 import overlap
 torch.set_num_threads(4);check_parent();p=read(PARENT/'LOCK.json');labels=read(p['labels']);sel=read(PARENT/'SELECTION.json');rows=[];results=[]
 for r in p['rows']:
  f=PARENT/r['split']/(stem(r)+'.pt');r={**r,'parent_file':str(f),'parent_sha':sha(f)};x=load(f);gt=labels[r['key']];fit=x['fits']['F'];v=np.array(gt['valid'],bool);obs=np.zeros(len(v),bool);obs[x['observed']]=True
  metric=lambda b:score(b,gt,x['frame_ids'],x['indices'],x['observed'],x['anchors'])
  before=metric(x['before']);path=[];q0=overlap(np.asarray(x['before']),np.asarray(gt['boxes']))
  for t in fit['path']:
   q=overlap(np.asarray(t['boxes']),np.asarray(gt['boxes']));delta=q-q0;damaged=v&(delta<-.05);runs=[];start=None
   for i,yes in enumerate(list(damaged)+[False]):
    if yes and start is None:start=i
    if not yes and start is not None:runs.append((start,i-1));start=None
   meta=load(r['original'])['input'] if not path else meta
   path.append(dict(step=t['step'],loss=t['loss'],metrics=metric(t['boxes']),per_frame_IoU=[float(z) if valid else None for z,valid in zip(q,v)],observed_sIoU=float(q[v&obs].mean()) if (v&obs).any() else None,damage_gt5pp_frames=int(damaged.sum()),valid_frames=int(v.sum()),longest_damaged_run=max([b-a+1 for a,b in runs] or [0]),longest_damaged_seconds=max([(x['frame_ids'][b]-x['frame_ids'][a])/meta['fps'] for a,b in runs] or [0.])))
  selected=path[fit['selected_step']];best=max(path,key=lambda t:t['metrics']['vIoU_corrected']);cfg=sel[r['cohort']]['S'];sf=x['fits'][f'S|rho{cfg["rho"]}|m{cfg["multiplier"]}'];sm=metric(sf['final']['boxes']);local=selected['metrics']['vIoU_corrected']-before['vIoU_corrected']
  orig=load(r['original']);native_time=score(fit['final']['boxes'],gt,x['frame_ids'],orig['predictions']['Frozen']['indices'],x['observed'],x['anchors'])
  refs=[]
  for a in x['anchors']:
   i=a['position'];ri=float(overlap(np.array([a['box']]),np.array([gt['boxes'][i]]))[0]) if v[i] else None
   refs.append(dict(position=i,frame_id=x['frame_ids'][i],GT_scorable=bool(v[i]),reference_GT_IoU=ri,Before_GT_IoU=float(q0[i]) if v[i] else None,reference_geometry_delta=ri-float(q0[i]) if v[i] else None,identity='unresolved_pending_official_track_and_visual_review'))
  results.append(dict(key=r['key'],cohort=r['cohort'],source=r['source'],split=r['split'],query=r['query'],Before=before,F=selected['metrics'],F10=path[-1]['metrics'],selected_step=fit['selected_step'],best_step_diagnostic=best['step'],regret=best['metrics']['vIoU_corrected']-selected['metrics']['vIoU_corrected'],local_delta=local,S_minus_F=sm['vIoU_corrected']-selected['metrics']['vIoU_corrected'],S_selected=sm,formal_indices=x['indices'],native_indices=orig['predictions']['Frozen']['indices'],same_F_boxes_native_time=native_time,anchor_count=len(refs),references=refs,frame_ids=x['frame_ids'],observed=x['observed'],path=path,process_class='no_current_adaptation' if not refs else ('utility_gain_identity_unresolved' if local>.001 else 'utility_harm_identity_unresolved' if local<-.001 else 'utility_neutral_identity_unresolved')))
  rows.append(r)
 mandatory={'hcstvg1_test':{'positive':[],'negative':['hcstvg1_test:000427']},'vidstg_test':{'positive':['vidstg_test:001346','vidstg_test:005989','vidstg_test:002289'],'negative':['vidstg_test:007297','vidstg_test:007024','vidstg_test:004537']}};panel=[]
 for c,groups in mandatory.items():
  for sign,keys in groups.items():
   selected=list(keys);sgn=1 if sign=='positive' else -1
   candidates=sorted([r for r in results if r['cohort']==c and sgn*r['local_delta']>.001 and r['key'] not in keys],key=lambda r:(0 if sgn*r['local_delta']>.05 else 1,digest(r['key'])))
   selected += [r['key'] for r in candidates[:4-len(selected)]]
   panel += [dict(key=k,sign=sign,mandatory=k in keys,selection_hash=digest(k)) for k in selected]
 write(OUT/'PROCESS_RAW.json',results);write(OUT/'PANEL_SELECTION.json',panel)
 write(OUT/'LOCK.json',dict(rows=rows,panel=panel,labels=p['labels'],labels_sha=p['labels_sha'],pins=p['pins'],parent_manifest_sha=sha(PARENT/'RESULT_MANIFEST.json'),protocol_sha=sha(ROOT/'protocols/c1_correction_mechanism_v1.md'),max_seconds=3600,max_bytes=2*1024**3,max_new_fits=62,created=time.time(),oracle_diagnostic=True,reserved72_used=False))
 status(OUT/'STATUS.json',dict(status='readback_complete_identity_review_pending',readback_sources=64,panel_sources=len(panel)))
 print([(r['key'],r['sign']) for r in panel])
def evidence():
 from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations
 from scripts.prepare_stvg_fullscale_v1 import xywh
 from vg_tta.box_stability_diagnostics_v1 import overlap
 p=read(OUT/'LOCK.json');labels=read(p['labels']);raw=load_vidor_annotations(ROOT/'downloads/vidor/validation-annotation.zip',None);evidence={}
 for r in p['rows']:
  x=original(r);g=labels[r['key']];o=load(r['original'])['input'];a=g['official_annotation'];record=raw[r['source']] if r['cohort']=='vidstg_test' else None
  target=a.get('target_id','HC_target');frames=[]
  cats={int(z['tid']):z['category'] for z in record['subject/objects']} if record else {'HC_target':r['subject']}
  for i,fid in enumerate(x['frame_ids']):
   tracks=[]
   if record:
    for z in record['trajectories'][fid]:
     b=z['bbox'];box=xywh([b['xmin'],b['ymin'],b['xmax']-b['xmin'],b['ymax']-b['ymin']],o['width'],o['height']);tracks.append(dict(tid=z['tid'],category=cats[z['tid']],box=box,generated=z.get('generated')))
   elif g['valid'][i]:tracks=[dict(tid=target,category=r['subject'],box=g['boxes'][i],generated=False)]
   frames.append(tracks)
  def assoc(b,i):
   tr=frames[i];scores=overlap(np.tile(b,(len(tr),1)),np.array([z['box'] for z in tr]).reshape(-1,4)) if tr else []
   return sorted([dict(tid=z['tid'],category=z['category'],IoU=float(q)) for z,q in zip(tr,scores)],key=lambda z:-z['IoU'])
  positions=[]
  for cap in x['captures']:
   i=cap['position'];probe=cap['probe'];positions.append(dict(position=i,frame_id=x['frame_ids'][i],accepted=probe['accepted'],winner=probe.get('admission_winner'),Before_association=assoc(np.asarray(x['before'][i]),i),candidates=[dict(index=j,candidate_id=probe['candidate_ids'][j],box=b.tolist(),score=float(probe['target_scores'][j]),track_association=assoc(b.numpy(),i)) for j,b in enumerate(probe['boxes'])]))
  evidence[r['key']]=dict(target_tid=target,categories=cats,official_annotation=a,frame_tracks=frames,positions=positions,scope='overlap associations are proposals, not verified candidate identities')
 write(OUT/'OFFICIAL_TRACK_EVIDENCE.json',evidence)
 print('EVIDENCE',len(evidence))
def render():
 from PIL import Image,ImageDraw,ImageFont
 from vg_tta.exact_frame_decode_audit_v2 import decode
 p=read(OUT/'LOCK.json');e=read(OUT/'OFFICIAL_TRACK_EVIDENCE.json');labels=read(p['labels']);(OUT/'review').mkdir(exist_ok=True)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 for item in p['panel']:
  r=next(z for z in p['rows'] if z['key']==item['key']);x=original(r);o=load(r['original']);frames,ids=decode(o['input']);ev=e[r['key']];caps=ev['positions'];tw,th=330,230;im=Image.new('RGB',(tw*4,60+len(caps)*th),'#161d25');draw=ImageDraw.Draw(im);draw.text((8,5),r['key']+' '+r['query'][:140],font=font,fill='white');draw.text((8,25),'GREEN=official target; RED=prediction/candidate; numbers=pool index',font=font,fill='white')
  for row,cap in enumerate(caps):
   pos=cap['position'];data=[('Before',x['before'][pos],None)]+[(f"c{z['index']} id{z['candidate_id']} score{z['score']:.3f}"+(' WIN' if z['index']==cap['winner'] else ''),z['box'],z) for z in cap['candidates']]
   for col,(title,b,z) in enumerate(data):
    tile=Image.fromarray(frames[pos]);ww,hh=tile.size;dd=ImageDraw.Draw(tile)
    def box(bb,color,width):
     cx,cy,w,h=map(float,bb);dd.rectangle([(cx-w/2)*ww,(cy-h/2)*hh,(cx+w/2)*ww,(cy+h/2)*hh],outline=color,width=width)
    for track in ev['frame_tracks'][pos]:
     if track['tid']==ev['target_tid']:box(track['box'],'#25ef52',3)
    box(b,'#ff273e',3);tile.thumbnail((tw,th-35));xx=col*tw;yy=60+row*th;im.paste(tile,(xx,yy+35));draw.text((xx+3,yy+2),f"p{pos} f{ids[pos]} {'A' if cap['accepted'] else 'REJ'} {title}",font=font,fill='white')
  im.save(OUT/'review'/(stem(r)+'.jpg'),quality=90)
  print('REVIEW',r['key'],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','evidence','render']);q=a.parse_args();globals()[q.phase]()
