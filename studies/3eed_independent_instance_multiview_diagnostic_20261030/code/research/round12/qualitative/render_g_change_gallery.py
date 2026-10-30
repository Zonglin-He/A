"""Private postscore/output-conditioned eightcase gallery from immutable R12 G.
No GT/taskscore, registration estimation, model, GPU or optimizer APIs.
"""
from pathlib import Path
import hashlib,json,sys,time,textwrap
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent;A=B/'research/round12/data_audit';E=B/'research/round12/experiment'
sys.path.insert(0,str(D));from render_input_gallery5 import font,visible_projected_points
sys.path.insert(0,str(B/'crossplatform/round7'));from camera_projection import CameraSpec,raw_to_model_points,project_model_points
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
G=B/'crossplatform/round12/feedback146_attempt001';M=B/'crossplatform/round12/masks501_attempt001'
ARMCOLOR=[(255,55,55),(30,220,255),(100,255,50)];ARMS=[0,2,3];NAMES=['source','single','MV'];EDGES=[(i,i^(1<<j)) for i in range(8) for j in range(3) if not i&(1<<j)]
def corners(box):return np.asarray(box[:3],float)+np.asarray([[(-1 if i&(1<<j)==0 else 1) for j in range(3)] for i in range(8)])*np.asarray(box[3:],float)/2

def make_camera(fr):
 fields={'image_intrinsic':fr['image_intrinsic'],'image_extrinsic':fr['image_extrinsic']}
 if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
 return CameraSpec.from_camera_fields(fr['platform'],fields,missing_distortion_policy='assume_zero_declared')

def load_observation(alias,fi,mt):
 fr=fi[alias];assert sha(fr['imagepath'])==fr['image_sha256']==mt[alias]['image_sha256'] and sha(fr['pointpath'])==fr['point_sha256']
 with Image.open(fr['imagepath']) as im:rgb=np.asarray(im.convert('RGB'))
 with np.load(M/(alias+'.npz'),allow_pickle=False) as z:masks=z['masks_bool'].copy();labels=z['class_ids'].copy();scores=z['scores'].copy()
 assert [hashlib.sha256(x.tobytes()).hexdigest() for x in masks]==mt[alias]['mask_sha256']
 raw=np.fromfile(fr['pointpath'],np.float32).reshape(-1,4)[:,:3];cam=make_camera(fr);pix,pdiag=visible_projected_points(raw_to_model_points(raw,fr['platform']),cam,(rgb.shape[1],rgb.shape[0]))
 return {'rgb':rgb,'masks':masks,'labels':labels,'scores':scores,'cam':cam,'pix':pix,'diag':pdiag,'frame':fr}

def mask_panel(obs,selected,title):
 rgb=obs['rgb'].astype(float).copy();mask=obs['masks'][selected] if selected is not None else np.zeros(rgb.shape[:2],bool);rgb[mask]=rgb[mask]*.6+np.asarray([50,190,255])*.4;im=Image.fromarray(rgb.astype(np.uint8));draw=ImageDraw.Draw(im)
 for x,y in obs['pix']:
  c=(0,255,100) if mask[y,x] else (255,135,0);draw.ellipse((x-1,y-1,x+1,y+1),fill=c)
 draw.rectangle((0,0,im.width,52),fill=(0,0,0));draw.text((6,4),title,fill='white',font=font(15));s='No selected matched mask' if selected is None else f'Sealed instance {selected}, COCO label {int(obs["labels"][selected])}, conf {float(obs["scores"][selected]):.3f} (prediction)';draw.text((6,28),s,fill='white',font=font(14));return im

def box_panel(obs,boxes,title,T_B_to_A=None):
 im=Image.fromarray(obs['rgb']);draw=ImageDraw.Draw(im);W,H=im.size
 for j,box in enumerate(boxes):
  xyz=corners(box)
  if T_B_to_A is not None:
   # Source model coordinates -> rawA -> inverse sealed T -> rawB -> modelB.
   raw=xyz.copy()
   if obs['frame']['platform']=='drone':raw[:,2]-=1.8
   T=np.linalg.inv(np.asarray(T_B_to_A,float));raw=raw@T[:3,:3].T+T[:3,3];xyz=raw.copy()
   if obs['frame']['platform']=='drone':xyz[:,2]+=1.8
  p=project_model_points(xyz,obs['cam'],(W,H));uv=p['uv'];depth=p['depth']
  for u,v in EDGES:
   if np.isfinite(uv[[u,v]]).all() and min(depth[u],depth[v])>.1:draw.line([tuple(uv[u]),tuple(uv[v])],fill=ARMCOLOR[j],width=3)
 draw.rectangle((0,0,W,52),fill=(0,0,0));draw.text((6,4),title,fill='white',font=font(15))
 for j,n in enumerate(NAMES):draw.text((6+140*j,28),n,fill=ARMCOLOR[j],font=font(15))
 return im

def main():
 planpath=E/'G_CHANGE_GALLERY_PLAN.json';plan=read(planpath);assert plan['renderer_sha256']==sha(__file__) and plan['n']==8 and plan['no_GT_or_task_scores_for_selection']
 for path,key in ((G/'SEALED.json','prediction_seal_sha256'),(M/'SEALED.json','mask_seal_sha256'),(A/'FRAME_INDEX_PRIVATE.json','frame_index_sha256'),(A/'ANCHOR146_GPU_INPUTS.json','manifest_sha256')):assert sha(path)==plan[key]
 gs=read(G/'SEALED.json');ms=read(M/'SEALED.json')
 for base,se in ((G,gs),(M,ms)):
  for f,h in se['files'].items():assert sha(base/f)==h
 tr={x['id']:x for x in read(G/'trace.json')};fi=read(A/'FRAME_INDEX_PRIVATE.json');mt={x['frame_alias']:x for x in read(M/'TRACE.json')};inputs={x['id']:x for x in read(A/'ANCHOR146_GPU_INPUTS.json')}
 out=D/'g_change_gallery8_attempt001';assert not out.exists();out.mkdir();files={};rows=[];start=time.perf_counter()
 for pick in plan['rows_PRIVATE']:
  sid=pick['id_PRIVATE'];trace=tr[sid];selected=trace['diagnostics']['selected_anchor_mask_index'];anchor=load_observation(trace['anchor_frame_alias'],fi,mt);assert selected is not None
  assert hashlib.sha256(anchor['masks'][selected].tobytes()).hexdigest()==trace['diagnostics']['anchor_mask_sha256']
  with np.load(G/(sid+'.npz'),allow_pickle=False) as z:boxes=z['arm_boxes_raw6'][ARMS].copy();changes=z['actual_changed_axes'][ARMS].copy();q=int(z['q_source'])
  assert np.array_equal(changes[1:],pick['changed_single_MV']) and changes[1:].any() and q==trace['q_source']
  ref=load_observation(pick['reference_frame_alias_PRIVATE'],fi,mt) if pick['reference_frame_alias_PRIVATE'] else None
  refindex=pick['reference_selected_mask_index'];T=None
  if pick['reference_verified']:
   attempt=next(v for v in trace['diagnostics']['neighbor_attempts'] if v['frame_alias']==pick['reference_frame_alias_PRIVATE']);assert attempt['verified'] and attempt['registration']['accepted'];T=attempt['registration']['T_raw_B_to_A'];assert refindex==attempt['neighbor_mask_index']
  panels=[mask_panel(anchor,selected,'ANCHOR: selected independent mask; green in-mask / orange outside'),box_panel(anchor,boxes,'ANCHOR: sealed fixed-q0 predictions (no GT)')]
  if ref:
   panels.append(mask_panel(ref,refindex,'REFERENCE: verified hypothesis' if pick['reference_verified'] else 'REFERENCE: candidate only; not verified'))
   if T is not None:panels.append(box_panel(ref,boxes,'REFERENCE: inverse SEALED T projection; no new fit',T))
   else:
    im=Image.fromarray(ref['rgb']);dd=ImageDraw.Draw(im);dd.rectangle((0,0,im.width,55),fill='black');dd.text((6,8),'No verified sealed T for this reference: boxes NOT transferred',fill='white',font=font(16));panels.append(im)
  else:panels.extend([Image.new('RGB',panels[0].size,(40,40,40))]*2)
  W,H=panels[0].size;page=Image.new('RGB',(2*W,2*H+195),(20,20,20))
  for k,im in enumerate(panels):page.paste(im,((k%2)*W,(k//2)*H))
  dd=ImageDraw.Draw(page);lines=[pick['case_alias']+f' | {trace["platform"]} | q0={q} fixed | single/MV axis-change={changes[1:].astype(int).tolist()}',f'Reference status: {pick["reference_status"]}; clock/synchronization/semantic identity unverified.', 'ALL output-changed union cases, postscore exploratory visualization; no task scores or GT read for selection/render.', 'Caption (input, not truth): '+inputs[sid]['caption']]
  y=2*H+8
  for line in lines:
   for segment in textwrap.wrap(line,width=240):dd.text((8,y),segment,fill='white',font=font(14));y+=22
  name=pick['case_alias']+'.jpg';page.save(out/name,quality=92);files[name]=sha(out/name);rows.append({'case_alias':pick['case_alias'],'platform':trace['platform'],'reference_verified':pick['reference_verified'],'reference_status':pick['reference_status'],'source_query_fixed':q==trace['q_source'],'changed_single_MV':changes[1:].tolist(),'anchor_selected_mask_sha256':trace['diagnostics']['anchor_mask_sha256'],'reference_mask_used':refindex is not None,'sealed_T_used':T is not None,'image_point_and_mask_hashes_checked':True})
 receipt={'state':'actual_private_R12_all8_changed_union_gallery_no_GT_taskscore','n':8,'selection':'All single-or-MV changed union; postscore/output-conditioned exploratory, not performancevalidation','plan_sha256':sha(planpath),'renderer_sha256':sha(__file__),'prediction_seal_sha256':sha(G/'SEALED.json'),'mask_seal_sha256':sha(M/'SEALED.json'),'rows':rows,'seconds':time.perf_counter()-start,'no_GT_taskscore_model_GPU_newregistration_or_optimizer':True};(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');files['RECEIPT.json']=sha(out/'RECEIPT.json');(out/'SEALED.json').write_text(json.dumps({'state':'sealed_private_R12_changed_union8_gallery','n':8,'plan_sha256':sha(planpath),'renderer_sha256':sha(__file__),'files':files},indent=2)+'\n');print(json.dumps({'n':8,'seal_sha256':sha(out/'SEALED.json'),'output':str(out)}))
if __name__=='__main__':main()
