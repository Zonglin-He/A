"""CPU-only post-seal attribution; never imports a model loader or fits parameters.

Private pixels, Query text and coordinates stay in artifacts. Public export contains
only anonymous scalar diagnostics and explicitly model-assisted visual judgments.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_dino_error_attribution32_v1'
JOINT=ROOT/'artifacts/stvg_joint_event_critic_v1'
OLD=ROOT/'artifacts/stvg_dino_rollback32_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load(p):
 import torch
 assert sha(p)==read(p.with_suffix('.receipt.json'))['sha256']
 return torch.load(p,map_location='cpu',weights_only=False)
def prepare():
 for b in [JOINT,OLD]:assert read(b/'CPU_COMPLETION.json')['count']==32
 a=read(JOINT/'PLAN_LOCK.json')['target'];b=read(OLD/'PLAN_LOCK.json')['records']
 assert [(r['alias'],r['dataset'],r['query_ordinal'])for r in a]==[(r['alias'],r['dataset'],r['query_ordinal'])for r in b]
 put(BASE/'PROTOCOL_LOCK.json',dict(status='locked',count=32,cohort='previously exposed HC16/Vid16; all retained',
  scope='saved-output CPU diagnostics and actual pixel review only',new_models=0,new_training=0,new_teacher_calls=0,
  contrasts={'Direct':'episodic Source to final Direct, fixed Native interval','OPD_current':'inherited Before to final P7-A, fixed projection interval'},
  support=['admitted observed GT frames','unobserved native-grid GT frames','benchmark interpolated GT frames'],
  harm='strict negative paired vIoU on fixed interval; dense-s harm additionally reported, no threshold selection',
  pixels='all32 sources, all four planned anchors, two largest absolute unobserved-native changes per method where GT exists',
  labels='model-assisted; identity from actual pixels plus full Query; low IoU is not identity; unknown/mixed retained',
  target_GT='offline diagnosis only after original prediction seals; no training, routing or method promotion',
  existing_512_panel_unchanged=True,
  inputs={str(b.relative_to(ROOT)):sha(b)for b in [JOINT/'PLAN_LOCK.json',JOINT/'POSTSEAL_RESULTS.json',JOINT/'SELECTION_BARRIER.json',OLD/'PLAN_LOCK.json',OLD/'POSTSEAL_RESULTS.json']}))
def run(render=False):
 import numpy as np,torch
 torch.set_num_threads(2)
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_decota_paper_main_v1 import read_row
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 assert (BASE/'PROTOCOL_LOCK.json').exists()
 records=read(JOINT/'PLAN_LOCK.json')['target'];oldresults=read(OLD/'POSTSEAL_RESULTS.json')
 oldidx={(r['case_id'],r['arm']):r['metrics']for r in oldresults['rows']}
 prior={r['alias']:r for r in read(JOINT/'POSTSEAL_RESULTS.json')['rows']}
 targets={ds:truths(ds,{'parents':[r['query_ordinal']for r in records if r['dataset']==ds]})for ds in ['hc2','vidstg']}
 rows=[];panels=[];checks=0
 for r in records:
  alias=r['alias'];ds=r['dataset'];ordinal=r['query_ordinal'];row=read_row(ds,ordinal)
  z=load(JOINT/'target'/(alias+'.pt'));p=load(OLD/'predictions'/(alias+'.pt'))
  assert z['expert']['anchors']==p['expert']['anchors']
  gt=targets[ds][0][ordinal];span=targets[ds][1][ordinal];ids=np.asarray(z['frame_ids'])
  assert list(ids)==row['frame_ids']
  anc=z['expert']['anchors']['single4'];positions=[a['position']for a in anc]
  entry=dict(source_alias=alias,dataset=ds,planned_anchors=4,admitted_anchors=len(anc),anchors=[],methods={})
  for a in anc:
   pos=a['position'];fid=int(ids[pos]);ref=xyxy(np.asarray(a['box']),row['input']['width'],row['input']['height'])
   entry['anchors'].append(dict(slot=z['expert']['positions4'].index(pos),GT_available=fid in gt,
    reference_GT_IoU=float(box_iou(ref,gt[fid]))if fid in gt else None))
  selected={}
  for method,pre,post,interval,oldarm in [
   ('Direct',z['boxes'][0].numpy(),z['boxes'][1].numpy(),z['native_interval'],None),
   ('OPD_current',p['fit']['before'].numpy(),p['fit']['final'].numpy(),p['arms']['Full_P7A']['interval'],'Full_P7A')]:
   tubes=[DenseTube(b,row,gt,span,clip=ds=='hc2')for b in [pre,post]]
   vals=[official(b,row,gt,span,interval,ds)for b in [pre,post]]
   for tube,v in zip(tubes,vals):
    assert max(abs(tube.score(interval)[k]-v[k])for k in ['v','t','s'])<2e-10;checks+=3
   if method=='Direct':
    for k in ['v','t','s']:assert abs(vals[1][k]-prior[alias]['values']['Direct_native'][k])<2e-10
   else:
    assert abs(vals[1]['v']-oldidx[alias,oldarm]['m_vIoU'])<2e-10
    assert abs(vals[0]['v']-oldidx[alias,'Before_projection']['m_vIoU'])<2e-10
   checks+=3
   diff=tubes[1].iou-tubes[0].iou;fids=tubes[0].fids
   obs=np.isin(fids,ids[positions]);native=np.isin(fids,ids)
   masks={'observed':obs,'unobserved_native':native&~obs,'benchmark_interpolated':~native}
   a,b=interval;g,h=span;active=(fids>=max(a,g))&(fids<min(b,h));den=max(max(b,h)-min(a,g),1)
   components={}
   for name,mask in masks.items():
    components[name]=dict(GT_frames=int(mask.sum()),inside_readout_frames=int((mask&active).sum()),
     pre_mean_IoU=float(tubes[0].iou[mask].mean())if mask.any()else None,
     post_mean_IoU=float(tubes[1].iou[mask].mean())if mask.any()else None,
     delta_mean_IoU=float(diff[mask].mean())if mask.any()else None,
     dense_s_contribution_pp=float(100*diff[mask].sum()/len(fids)),
     v_contribution_pp=float(100*diff[mask&active].sum()/den),
     gross_loss_v_pp=float(100*np.minimum(diff[mask&active],0).sum()/den),
     gross_gain_v_pp=float(100*np.maximum(diff[mask&active],0).sum()/den))
   assert abs(sum(c['v_contribution_pp']for c in components.values())-100*(vals[1]['v']-vals[0]['v']))<2e-8
   assert abs(sum(c['dense_s_contribution_pp']for c in components.values())-100*(vals[1]['s']-vals[0]['s']))<2e-8
   checks+=2
   beforexy=xyxy(pre,row['input']['width'],row['input']['height']);afterxy=xyxy(post,row['input']['width'],row['input']['height'])
   agreement=[float(np.mean([box_iou(b[a['position']],xyxy(np.asarray(a['box']),row['input']['width'],row['input']['height']))for a in anc]))if anc else None for b in [beforexy,afterxy]]
   anchor_changes=[]
   for ai,a0 in enumerate(anc):
    fid=int(ids[a0['position']]);idx=np.flatnonzero(fids==fid)
    anchor_changes.append(dict(slot=entry['anchors'][ai]['slot'],GT_available=bool(len(idx)),
     pre_GT_IoU=float(tubes[0].iou[idx[0]])if len(idx)else None,post_GT_IoU=float(tubes[1].iou[idx[0]])if len(idx)else None))
   entry['methods'][method]=dict(pre=vals[0],post=vals[1],delta_v_pp=100*(vals[1]['v']-vals[0]['v']),
    delta_s_pp=100*(vals[1]['s']-vals[0]['s']),DINO_agreement=agreement,components=components,anchors=anchor_changes,
    harmful_v=vals[1]['v']<vals[0]['v']-1e-12,harmful_s=vals[1]['s']<vals[0]['s']-1e-12,
    reference_agreement_improved=bool(anc and agreement[1]>agreement[0]+1e-12))
   # Actual native-grid positions only; interpolated benchmark pixels are not fabricated.
   candidates=[int(np.flatnonzero(ids==f)[0])for f in fids[native&~obs]]
   lookup={int(f):float(d)for f,d in zip(fids,diff)}
   selected[method]=sorted(candidates,key=lambda j:(-abs(lookup[int(ids[j])]),j))[:2]
  rows.append(entry)
  if render:
   panels.append(render_case(r,row,z,p,gt,selected))
 summary={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];summary[ds]={}
  for method in ['Direct','OPD_current']:
   mm=[r['methods'][method]for r in rr]
   summary[ds][method]=dict(count=len(mm),harmful_v=sum(x['harmful_v']for x in mm),harmful_s=sum(x['harmful_s']for x in mm),
    mean_delta_v_pp=float(np.mean([x['delta_v_pp']for x in mm])),mean_delta_s_pp=float(np.mean([x['delta_s_pp']for x in mm])),
    agreement_up_v_down=sum(x['harmful_v']and x['reference_agreement_improved']for x in mm),
    components={k:dict(v_pp=float(np.mean([x['components'][k]['v_contribution_pp']for x in mm])),s_pp=float(np.mean([x['components'][k]['dense_s_contribution_pp']for x in mm])))for k in ['observed','unobserved_native','benchmark_interpolated']})
 put(BASE/'NUMERIC_RESULTS.json',dict(status='complete',count=32,checks=checks,summary=summary,rows=rows))
 if render:put(BASE/'PIXEL_PANEL_MANIFEST.json',dict(status='rendered_not_yet_reviewed',count=len(panels),panels=panels))
 print(json.dumps(dict(checks=checks,summary=summary,harmful_union=[r['source_alias']for r in rows if any(x['harmful_v']for x in r['methods'].values())])))
def render_case(r,row,z,p,gt,selected):
 import numpy as np,textwrap
 from PIL import Image,ImageDraw,ImageFont
 from scripts.run_decota_paper_main_v1 import frames_for
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 frames,ids,_=frames_for(r['dataset'],row,collections.OrderedDict())
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
 canvas=Image.new('RGB',(1600,1060),'white');d=ImageDraw.Draw(canvas)
 query=row['input']['caption'];title=r['alias']+' '+r['dataset']+' PRIVATE; model-assisted diagnostic review\n'+'\n'.join(textwrap.wrap(query,145))
 d.multiline_text((12,8),title,fill='black',font=font)
 d.text((12,92),'ANCHORS: orange DINO / cyan GT. CHANGES: red before / green after / cyan GT. No GT outside annotation = unknown.',fill='black',font=font)
 jobs=[('anchor',i,pos)for i,pos in enumerate(z['expert']['positions4'])]
 jobs +=[(method,i,pos)for method in ['Direct','OPD_current']for i,pos in enumerate(selected[method])]
 for cell,(method,slot,pos)in enumerate(jobs):
  fid=int(ids[pos]);im=Image.fromarray(frames[pos]);ow,oh=im.size;im.thumbnail((396,420));w,h=im.size
  x=(cell%4)*400+(400-w)//2;y=130+(cell//4)*460
  canvas.paste(im,(x,y));dr=ImageDraw.Draw(canvas)
  def box(bb,col):
   a=np.asarray(bb,float)*np.array([w/ow,h/oh,w/ow,h/oh]);dr.rectangle((x+a[0],y+a[1],x+a[2],y+a[3]),outline=col,width=3)
  if fid in gt:box(gt[fid],'cyan')
  if method=='anchor':
   aa=[a for a in z['expert']['anchors']['single4']if a['position']==pos]
   if aa:box(xyxy(np.asarray(aa[0]['box']),ow,oh),'orange')
   text=f'anchor {slot}; GT {fid in gt}; admitted {bool(aa)}'
  else:
   pre,post=(z['boxes'][0],z['boxes'][1])if method=='Direct'else(p['fit']['before'],p['fit']['final'])
   box(xyxy(pre[pos].numpy(),ow,oh),'red');box(xyxy(post[pos].numpy(),ow,oh),'lime')
   text=f'{method} nonobserved native #{slot}'
  dr.text(((cell%4)*400+4,y+422),text,fill='black',font=font)
 path=BASE/'private_pixels'/(r['alias']+'.jpg');path.parent.mkdir(exist_ok=True);canvas.save(path,quality=94)
 return dict(source_alias=r['alias'],path=str(path.relative_to(ROOT)),sha256=sha(path),cells=len(jobs),actual_frames=True,query_text_private=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','analyze']);a.add_argument('--render',action='store_true');args=a.parse_args()
 prepare()if args.stage=='prepare'else run(args.render)
