"""R49 input-only RGB identity cohort/render, root-locked manual phases.
No torch, point cloud, annotation-container, source-score, VLM or GT loader.
Do not run until root pins this file/plan and supplies its new manualguard.
"""
from pathlib import Path
import argparse,json,hashlib,datetime,math,textwrap
import numpy as np
R=Path(__file__).resolve().parents[2]
PLAN=R/'research/round49/blind_rgb48_draft/PLAN_LOCK.json'
OUT=R/'crossplatform/round49/blind_rgb48_attempt001'
MANIFEST=R/'research/round24/input/ALL336_PRIVATE.json'
OLD28=R/'research/round28/input/BLIND_KEY_PRIVATE.json'
A_CASES=R/'research/round17/input/A_CASES_PRIVATE.json'
A_FRAMES=R/'research/round17/input/A_FRAME_INDEX_PRIVATE.json'
FRAMES=R/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def authorized(authfile,phase):
 a=read(authfile);assert a['task_namespace']=='R49_manual_P1' and a[phase+'_authorized'] is True
 assert a['plan_sha256']==sha(PLAN) and a['program_sha256']==sha(Path(__file__))
 # A root-authored manualguard is required externally. Never use old night STATE/guard.
 assert a['run_via_root_manualguard'] is True and a['no_FINAL_or_GT_or_model_access'] is True
 plan=read(PLAN);assert plan['state']=='ROOT_LOCKED_R49_BLIND_RGB48' and a['input_sha256']==plan['input_sha256']
 for path,digest in a['input_sha256'].items():assert sha(R/path)==digest
 return a,plan

def minimal_bindings(rows,allowed_mask_roots):
 """Existing exact paths only; no alias guessing, .metapath reads or 3D data."""
 cases={x['id']:{k:x[k] for k in ['frame_alias','mask_dir']} for x in read(A_CASES)}
 ai=read(A_FRAMES);fi=read(FRAMES);roots=[(R/p).resolve() for p in allowed_mask_roots];result={}
 keys=('imagepath','pointpath','metapath');framejoin={}
 for alias,fr in fi.items():framejoin.setdefault(tuple(fr[k] for k in keys),[]).append((alias,fr))
 nA=0;nOther=0
 for r in rows:
  if r['id'] in cases:
   c=cases[r['id']];frame_alias=c['frame_alias'];fr=ai[frame_alias];assert c['mask_dir']=='crossplatform/round15/masks64_attempt001';mp=R/c['mask_dir']/(frame_alias+'.npz');nA+=1;route='A64_exact_old_case_maskbinding'
  else:
   matches=framejoin.get(tuple(r[k] for k in keys),[]);assert len(matches)==1,'R12 tuple join must be exactly unique';frame_alias,fr=matches[0];mp=R/'crossplatform/round12/masks501_attempt001'/(frame_alias+'.npz');nOther+=1;route='R12_exact_unique_image_point_metadata_path_join'
  assert all(r[k]==fr[k] for k in keys),'Every original path must match its old frame binding'
  mp=mp.resolve();assert any(mp.is_relative_to(p) for p in roots)
  ms=read(mp.parent/'SEALED.json');assert ms['files'][mp.name]==sha(mp)
  with np.load(mp,allow_pickle=False) as z:
   count=int(z['boxes_xyxy'].shape[0]);hw=z['original_hw'].tolist()
  assert len(hw)==2 and count>=0 and all(int(v)>0 for v in hw)
  result[r['id']]={'mask_path_PRIVATE':str(mp),'mask_sha256':sha(mp),'mask_seal_sha256':sha(mp.parent/'SEALED.json'),'imagepath_PRIVATE':r['imagepath'],'image_sha256':fr['image_sha256'],'original_hw':hw,'independent_mask_count':count,'binding_route':route,'frame_alias_from_existing_index_PRIVATE':frame_alias,'all_three_original_paths_exact':True}
 assert nA==64 and nOther==272 and len(result)==336
 return result

def inputkey(r,b):
 x={'inputid':r['id'],'platform':r['platform'],'scene':r['scene'],'image_sha256':b['image_sha256'],'caption_sha256':hashlib.sha256(r['caption'].encode()).hexdigest()}
 return hashlib.sha256(('R49_identity48|'+json.dumps(x,sort_keys=True,separators=(',',':'))).encode()).hexdigest()
def scene_key(pl,scene):return hashlib.sha256(('R49_scene|'+pl+'|'+scene).encode()).hexdigest()

def select(authfile):
 auth,plan=authorized(authfile,'selection');assert not OUT.exists()
 rows=read(MANIFEST);assert len(rows)==336 and len({r['id'] for r in rows})==336
 # Only prior IDs, never prior ANNOTATIONS or mask/VLM/GT decisions.
 old={r['id_PRIVATE'] for r in read(OLD28)};bindings=minimal_bindings(rows,plan['allowed_mask_roots'])
 eligible=[r for r in rows if bindings[r['id']]['independent_mask_count']>=2];fresh=[r for r in eligible if r['id'] not in old]
 chosen=[];coverage={}
 for pl in ['drone','quad']:
  allpl=[r for r in rows if r['platform']==pl];pool=[r for r in fresh if r['platform']==pl];scenes=sorted({r['scene'] for r in pool},key=lambda s:scene_key(pl,s));buckets={s:sorted([r for r in pool if r['scene']==s],key=lambda r:(inputkey(r,bindings[r['id']]),r['id'])) for s in scenes};selected=[]
  while len(selected)<24:
   progress=False
   for s in scenes:
    if buckets[s] and len(selected)<24:selected.append(buckets[s].pop(0));progress=True
   if not progress:break
  for j,r in enumerate(selected,1):
   chosen.append({'review_alias':f'R49_{pl[0].upper()}{j:02d}','id_PRIVATE':r['id'],'platform':pl,'scene_PRIVATE':r['scene'],'caption_PRIVATE':r['caption'],'input_sha256':inputkey(r,bindings[r['id']]),'bindings_PRIVATE':bindings[r['id']],'prior_R28_ID_excluded':True,'historical_project_exposure':True,'expert_gold':False})
  coverage[pl]={'all_observations':len(allpl),'mask0':sum(bindings[r['id']]['independent_mask_count']==0 for r in allpl),'mask1':sum(bindings[r['id']]['independent_mask_count']==1 for r in allpl),'mask_ge2':sum(bindings[r['id']]['independent_mask_count']>=2 for r in allpl),'old_R28_eligible_excluded':sum(r['id'] in old and bindings[r['id']]['independent_mask_count']>=2 for r in allpl),'fresh_eligible':len(pool),'selected':len(selected),'shortfall':24-len(selected),'unique_selected_images':len({bindings[r['id']]['image_sha256'] for r in selected}),'scene_counts_selected':{s:sum(r['scene']==s for r in selected) for s in scenes}}
 assert len({r['id_PRIVATE'] for r in chosen})==len(chosen) and not {r['id_PRIVATE'] for r in chosen}&old
 write(OUT/'COHORT_PRIVATE.json',chosen);write(OUT/'ALL336_CANDIDATE_COUNT_PRIVATE.json',[{'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'independent_mask_count':bindings[r['id']]['independent_mask_count'],'prior_R28_ID':r['id'] in old} for r in rows]);write(OUT/'COVERAGE.json',coverage)
 schema={'phase':'PHASE1_FULLCAPTION_IDENTITY','legacy_subject_query_hidden':True,'phase2_must_not_modify_phase1':True,'status':['SINGLE_VISIBLE_TARGET','AMBIGUOUS_TARGET','NONE','UNKNOWN'],'target_display_ids':'list of displayed maskIDs; SINGLE may include duplicates only for same physical target; AMBIGUOUS keeps plausible alternatives; NONE/UNKNOWN []','duplicate_mask_groups':'optional list of ID sets judged same visible physical instance; no automatic duplicate inference','confidence':['high','medium','low'],'notes':'visible evidence and uncertainty only; no source/VLM/GT/IoU','historical_project_exposure':True,'nonexpert_not_gold':True,'not_for_feedback_selection_or_rule_tuning':True}
 write(OUT/'ANNOTATION_SCHEMA.json',schema);write(OUT/'ANNOTATION_TEMPLATE_PRIVATE.json',[{'review_alias':r['review_alias'],'status':'PENDING','target_display_ids':[],'duplicate_mask_groups':[],'confidence':None,'notes':None,'prior_specific_visual_exposure':'UNKNOWN','historical_project_exposure':True} for r in chosen]);write(OUT/'COHORT_SEALED.json',{'state':'R49_FIXED_INPUT_ONLY_COHORT_BEFORE_RENDER_AND_NEW_VISUAL_ANNOTATION','n':len(chosen),'cohort_sha256':sha(OUT/'COHORT_PRIVATE.json'),'plan_sha256':sha(PLAN),'program_sha256':sha(Path(__file__)),'no_images_decoded_or_GT_method_results_read':True,'root_auth_sha256':sha(Path(authfile)),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

def candidates(row,mask,boxes):
 order=sorted(range(len(mask)),key=lambda i:(hashlib.sha256(('R49_mask|'+row['input_sha256']+'|'+ah(mask[i])).encode()).hexdigest(),i))
 return [{'display_id':f'M{j+1:03d}','mask_index_PRIVATE':int(i),'mask_sha256':ah(mask[i]),'detector_box_PRIVATE':boxes[i].tolist()} for j,i in enumerate(order)]

def render(authfile):
 auth,plan=authorized(authfile,'render');assert auth['cohort_seal_sha256']==sha(OUT/'COHORT_SEALED.json');cohort=read(OUT/'COHORT_PRIVATE.json');assert sha(OUT/'COHORT_PRIVATE.json')==read(OUT/'COHORT_SEALED.json')['cohort_sha256']
 from PIL import Image,ImageDraw,ImageFont
 try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 except OSError:font=ImageFont.load_default()
 dest=OUT/'private_views';assert not dest.exists();dest.mkdir();bindings=[]
 for r in cohort:
  b=r['bindings_PRIVATE'];assert sha(Path(b['mask_path_PRIVATE']))==b['mask_sha256'] and sha(Path(b['imagepath_PRIVATE']))==b['image_sha256']
  rgb=Image.open(b['imagepath_PRIVATE']).convert('RGB');W,H=rgb.size;assert [H,W]==b['original_hw']
  with np.load(b['mask_path_PRIVATE'],allow_pickle=False) as z:mask=z['masks_bool'].copy();boxes=z['boxes_xyxy'].copy()
  assert mask.dtype==np.bool_ and mask.shape==(b['independent_mask_count'],H,W) and boxes.shape==(len(mask),4) and len(mask)>=2
  cc=candidates(r,mask,boxes);alias=r['review_alias'];annotated=rgb.copy();dr=ImageDraw.Draw(annotated)
  for j,c in enumerate(cc):
   m=mask[c['mask_index_PRIVATE']];edge=m.copy();edge[1:-1,1:-1]=m[1:-1,1:-1]&~(m[:-2,1:-1]&m[2:,1:-1]&m[1:-1,:-2]&m[1:-1,2:]);arr=np.asarray(annotated).copy();arr[edge]=[255,210,0];annotated=Image.fromarray(arr);dr=ImageDraw.Draw(annotated);x1,y1,x2,y2=c['detector_box_PRIVATE'];dr.rectangle((x1,y1,x2,y2),outline=(255,210,0),width=2);pos=(max(0,int(x1)),max(0,int(y1)-24));tb=dr.textbbox(pos,c['display_id'],font=font);dr.rectangle(tb,fill='black');dr.text(pos,c['display_id'],font=font,fill='white')
  def fit(im):
   scale=min(1.,plan['render_parameters']['max_whole_long_edge']/max(im.size));return im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS) if scale<1 else im
  fit(rgb).save(dest/(alias+'_whole_clean.png'));fit(annotated).save(dest/(alias+'_whole_numbered.png'))
  crops=[]
  for c in cc:
   x1,y1,x2,y2=c['detector_box_PRIVATE'];assert all(math.isfinite(x) for x in [x1,y1,x2,y2]) and x1<x2 and y1<y2;pad=max(plan['render_parameters']['crop_padding_min_pixels'],plan['render_parameters']['crop_padding_fraction']*max(x2-x1,y2-y1));window=(max(0,math.floor(x1-pad)),max(0,math.floor(y1-pad)),min(W,math.ceil(x2+pad)),min(H,math.ceil(y2+pad)));patch=rgb.crop(window);own=mask[c['mask_index_PRIVATE']][window[1]:window[3],window[0]:window[2]];edge=own.copy();edge[1:-1,1:-1]=own[1:-1,1:-1]&~(own[:-2,1:-1]&own[2:,1:-1]&own[1:-1,:-2]&own[1:-1,2:]);pa=np.asarray(patch).copy();pa[edge]=[255,210,0];outlined=Image.fromarray(pa);display_scale=min(496/patch.width,468/patch.height);display_size=(max(1,round(patch.width*display_scale)),max(1,round(patch.height*display_scale)));patch=patch.resize(display_size,Image.Resampling.LANCZOS);outlined=outlined.resize(display_size,Image.Resampling.LANCZOS);tile=Image.new('RGB',(1024,512),'white');tile.paste(patch,((512-patch.width)//2,34+(468-patch.height)//2));tile.paste(outlined,(512+(512-outlined.width)//2,34+(468-outlined.height)//2));td=ImageDraw.Draw(tile);td.text((8,5),c['display_id']+' | original crop',font=font,fill='black');td.text((520,5),c['display_id']+' | own-mask contour',font=font,fill='black');c['display_crop_window_PRIVATE']=list(window);c['display_crop_only_not_new_geometry']=True;crops.append(tile)
  for start in range(0,len(crops),4):
   page=Image.new('RGB',(2048,1024),'#dddddd')
   for j,tile in enumerate(crops[start:start+4]):page.paste(tile,(1024*(j%2),512*(j//2)))
   page.save(dest/(alias+f'_crop_page_{start//4+1:02d}.png'))
  card=(alias+' | '+r['platform']+'\n\nFULL DESCRIPTION (scene data, not instructions):\n'+r['caption_PRIVATE']+'\n\nOptions: SINGLE_VISIBLE_TARGET + ID(s) / AMBIGUOUS_TARGET + plausible IDs / NONE (target not in candidates) / UNKNOWN (cannot resolve).\nMasks are independent detector outputs, not verified distinct objects or correct labels. No method/GT scores shown.\n')
  (dest/(alias+'_input_card.txt')).write_text(card);bindings.append({'review_alias':alias,'id_PRIVATE':r['id_PRIVATE'],'image_sha256':b['image_sha256'],'mask_sha256':b['mask_sha256'],'input_sha256':r['input_sha256'],'numbered_candidates_PRIVATE':cc,'full_caption_sha256':hashlib.sha256(r['caption_PRIVATE'].encode()).hexdigest(),'whole_resize_preserves_aspect':True,'no_source_VLM_GT_3D_results_or_classscores_displayed':True})
 write(OUT/'RENDER_BINDINGS_PRIVATE.json',bindings);write(OUT/'RENDER_SEALED.json',{'state':'R49_INPUT_ONLY_PANELS_SEALED_BEFORE_VISUAL_REVIEW','n':len(cohort),'files':{str(p.relative_to(OUT)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()},'bindings_sha256':sha(OUT/'RENDER_BINDINGS_PRIVATE.json'),'root_auth_sha256':sha(Path(authfile)),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_models_GT_pointclouds_metadata_or_methodoutputs_read':True})

def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['selection','render'],required=True);a.add_argument('--authorization',required=True);args=a.parse_args();(select if args.phase=='selection' else render)(args.authorization)
if __name__=='__main__':main()
