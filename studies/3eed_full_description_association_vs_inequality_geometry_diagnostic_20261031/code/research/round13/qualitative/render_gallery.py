"""Post-score R13 descriptive PRIVATE gallery. No GT metadata/index or new IoU.
Select fixed extreme/fixed regression cases, lock plan first, then render sealed
predictions on original RGB. No model, optimizer, G, registration or new scores.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,sys,textwrap
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'crossplatform/round7'))
from camera_projection import CameraSpec,project_model_points
COLORS={'source':'#00cbe6','A_full_consensus':'#ff9d19','B_inequality_full':'#39ee61','B_old_surface_equality':'#ff55d8'}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def verify(directory):
 seal=read(directory/'SEALED.json')
 for rel,h in seal['files'].items():
  p=Path(rel)
  if p.is_absolute() or '..' in p.parts or sha(directory/p)!=h:raise ValueError('Bad sealed member')
 return sha(directory/'SEALED.json')
def paths():
 return {'scores':ROOT/'research/round13/experiment/scored116_attempt001',
   'A':ROOT/'crossplatform/round13/association116_attempt001',
   'B':ROOT/'crossplatform/round13/geometry116_attempt001',
   'composite':ROOT/'crossplatform/round13/direct116_attempt001'}
def choose(rows,cases):
 alias={c['id']:c['case_alias'] for c in cases};byalias={alias[r['id']]:r for r in rows}
 def delta(r,arm):return r['arms'][arm]['iou']-r['arms']['source']['iou']
 def extreme(platform,arm,positive):
  rr=[r for r in rows if r['case_role']=='new_content108' and r['platform']==platform and (delta(r,arm)>0 if positive else delta(r,arm)<0)]
  return sorted(rr,key=lambda r:((-delta(r,arm) if positive else delta(r,arm)),alias[r['id']]))[0] if rr else None
 rules=[('new_A_largest_positive_Drone',extreme('drone','A_full_consensus',True)),
        ('new_A_largest_positive_Quad',extreme('quad','A_full_consensus',True)),
        ('new_A_largest_negative_Quad',extreme('quad','A_full_consensus',False)),
        ('new_B_largest_positive_Quad',extreme('quad','B_inequality_full',True)),
        ('new_B_largest_negative_Quad',extreme('quad','B_inequality_full',False))]
 negatives=[r for r in rows if r['case_role']=='new_content108' and r['platform']=='drone' and delta(r,'B_inequality_full')<0]
 if len(negatives)!=1 or alias[negatives[0]['id']]!='new_00015':raise ValueError('Predeclared unique negative Drone rule mismatch; do not silently replace')
 rules += [('new_B_unique_negative_Drone_new_00015',negatives[0]),
           ('fixed_regression_00002',byalias['regression_00002']),
           ('fixed_regression_00006',byalias['regression_00006'])]
 selected=[];seen=set();duplicate=[];missing=[]
 for reason,row in rules:
  if row is None:missing.append(reason);continue
  if row['id'] in seen:duplicate.append({'rule':reason,'alias':alias[row['id']]});continue
  seen.add(row['id']);selected.append({'id_PRIVATE':row['id'],'alias':alias[row['id']],
      'selection_rule':reason,'platform':row['platform'],'case_role':row['case_role'],
      'source_iou_sealed':row['arms']['source']['iou'],
      'arm_iou_sealed':{a:row['arms'][a]['iou'] for a in COLORS},
      'arm_delta_iou_sealed':{a:delta(row,a) for a in COLORS},
      'arm_status_sealed':{a:row['arms'][a]['status'] for a in COLORS}})
 return selected,duplicate,missing


def make_plan(planpath,output):
 pp=paths();gplan=read(pp['B']/'registered_plan.json')
 inp={k:Path(gplan[k]) for k in ('manifest','cases','frame_index')}
 seals={k:verify(v) for k,v in pp.items()}
 result=read(pp['scores']/'RESULTS_PRIVATE.json');cases=read(inp['cases'])
 selected,duplicates,missing=choose(result['rows_PRIVATE'],cases)
 plan={'state':'locked_R13_postscore_descriptive_gallery_before_render','qualitative_postscore_permitted':True,
   'GT_metadata_or_scoring_index_read_permitted':False,'new_IoU_optimization_model_G_or_registration_permitted':False,
   'source_GT_box_drawn':False,'blind_or_identity_truth_claimed':False,
   'selection':'Fixed signed extrema within new108/platform, source IoU reference; alias ascending ties; fixed regression2/6; deduplicate, no replacement.',
   'program_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
      'crossplatform/round7/camera_projection.py':sha(ROOT/'crossplatform/round7/camera_projection.py')},
   'directories':{k:str(v) for k,v in pp.items()},'seal_sha256':seals,
   'inputs':{k:{'path':str(v),'sha256':sha(v)} for k,v in inp.items()},
   'results_private_sha256':sha(pp['scores']/'RESULTS_PRIVATE.json'),
   'output':str(output.resolve()),'selected':selected,'duplicate_rules_not_replaced':duplicates,'missing_rules_not_replaced':missing,
   'planned_n':len(selected),'all_raw_media_caption_coordinates_remain_private':True,
   'projection':'R7 model coordinates; Drone calibrated E@undo_z exactly once; no provided pose',
   'render_layout':'2x2: prediction overlay; sealed numbering panel and chosen A mask; independent B mask/support; zoom overlay. No GT box.',
   'size_and_center_effect_scope':'A selects original candidate; B keeps q0 and changes bounded prior-regularized geometry. Visual overlap does not certify task identity.'}
 if planpath.exists():raise FileExistsError(planpath)
 dump(planpath,plan);print(json.dumps({'planned_n':len(selected),'plan_sha256':sha(planpath),'aliases':[s['alias'] for s in selected]}))


def box_projection(box,camera,wh):
 b=np.asarray(box,float);signs=np.array(list(itertools.product((-1.,1.),repeat=3)))
 points=b[:3]+signs*b[3:]/2
 return project_model_points(points,camera,wh)

def draw_box(ax,box,camera,wh,color,label,width=1.3,style='-'):
 pr=box_projection(box,camera,wh)
 if not np.isfinite(pr['uv']).all() or np.any(pr['depth']<=.1):
  return None
 uv=pr['uv']
 for i,j in itertools.combinations(range(8),2):
  if bin(i^j).count('1')==1:ax.plot(uv[[i,j],0],uv[[i,j],1],color=color,lw=width,ls=style,alpha=.9)
 ax.plot([],[],color=color,lw=width,label=label,ls=style)
 return uv

def overlay_mask(ax,mask,color,alpha=.12):
 if mask is None or not mask.any():return
 import matplotlib.colors as colors
 rgba=np.zeros((*mask.shape,4));rgba[mask]=(*colors.to_rgb(color),alpha);ax.imshow(rgba)
 ax.contour(mask.astype(float),levels=[.5],colors=[color],linewidths=.8,alpha=.8)


def render(planpath,expected):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 if sha(planpath)!=expected:raise ValueError('Plan hash mismatch')
 plan=read(planpath)
 assert plan['qualitative_postscore_permitted'] and not plan['GT_metadata_or_scoring_index_read_permitted']
 assert not plan['new_IoU_optimization_model_G_or_registration_permitted']
 for rel,h in plan['program_sha256'].items():
  if sha(ROOT/rel)!=h:raise ValueError('Renderer program changed')
 pp={k:Path(v) for k,v in plan['directories'].items()}
 for k,directory in pp.items():
  if verify(directory)!=plan['seal_sha256'][k]:raise ValueError('Source evidence changed')
 for entry in plan['inputs'].values():
  if sha(entry['path'])!=entry['sha256']:raise ValueError('Input-only join changed')
 manifest=read(plan['inputs']['manifest']['path']);cases=read(plan['inputs']['cases']['path'])
 frames=read(plan['inputs']['frame_index']['path'])
 byid={r['id']:r for r in manifest};ci={r['id']:r for r in cases}
 output=Path(plan['output'])
 if output.exists():raise FileExistsError(output)
 output.mkdir(parents=True)
 index=[]
 for selection in plan['selected']:
  sid=selection['id_PRIVATE'];alias=selection['alias'];row=byid[sid];case=ci[sid];fr=frames[case['frame_alias']]
  if sha(row['imagepath'])!=fr['image_sha256']:raise ValueError('Original image hash mismatch')
  # Input-only camera fields, never an annotation container or scoring index.
  fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
  if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
  camera=CameraSpec.from_camera_fields(row['platform'],fields,missing_distortion_policy='assume_zero_declared')
  camera.validated();rgb=np.array(Image.open(row['imagepath']).convert('RGB'),copy=True);H,W=rgb.shape[:2]
  with np.load(pp['composite']/(sid+'.npz'),allow_pickle=False) as z:
   names=z['arm_names'].tolist();boxes=z['arm_boxes_raw6'].copy();queries=z['arm_query_index'].copy();scores=z['source_text_scores256'].copy();q0=int(z['q_source'])
  b=np.load(pp['B']/(sid+'.npz'),allow_pickle=False)
  try:points=b['B_support_model_points'].copy();bmask_index=int(b['selected_mask_index'])
  finally:b.close()
  atrace=read(pp['A']/'cases'/(alias+'.json'));amask_index=atrace['consensus']['full']['selected_mask_index']
  maskdir=ROOT/case['mask_dir'];ms=read(maskdir/'SEALED.json');mp=maskdir/(case['frame_alias']+'.npz')
  if sha(mp)!=ms['files'][mp.name]:raise ValueError('Independent mask cache changed')
  with np.load(mp,allow_pickle=False) as z:masks=z['masks_bool'].copy()
  amask=masks[amask_index] if amask_index is not None else None
  bmask=masks[bmask_index] if bmask_index>=0 else None
  # Existing sealed panel, no newly rendered candidate ordering or model call.
  panelpath=pp['A']/'panels_PRIVATE'/(alias+'__order0.png')
  panel=np.array(Image.open(panelpath).convert('RGB'),copy=True)
  fig,axs=plt.subplots(2,2,figsize=(18,13),dpi=140)
  alluv=[]
  for ax in axs.ravel():ax.axis('off')
  axs[0,0].imshow(rgb)
  for name,color in COLORS.items():
   j=names.index(name);delta=selection['arm_delta_iou_sealed'][name]*100
   label=f'{name} q={queries[j]} ΔIoU={delta:+.3f}pp'
   uv=draw_box(axs[0,0],boxes[j],camera,(W,H),color,label,1.7 if name=='source' else 1.25,'--' if name=='B_old_surface_equality' else '-')
   if uv is not None:alluv.append(uv)
  axs[0,0].set_xlim(0,W);axs[0,0].set_ylim(H,0)
  axs[0,0].legend(loc='upper left',fontsize=8,facecolor='black',labelcolor='white',framealpha=.72)
  axs[0,0].set_title('Sealed selected 3D predictions projected on RGB — no GT boxes',fontsize=10)
  axs[0,1].imshow(panel);overlay_mask(axs[0,1],amask,'#ffe600',.12)
  axs[0,1].set_xlim(0,W);axs[0,1].set_ylim(H,0)
  axs[0,1].set_title(f'Sealed order0 panel; A full-consensus mask={amask_index}; not identity truth',fontsize=10)
  axs[1,0].imshow(rgb);overlay_mask(axs[1,0],bmask,'#00e8df',.14)
  if len(points):
   pr=project_model_points(points,camera,(W,H));take=pr['in_image'];uv=pr['uv'][take]
   axs[1,0].scatter(uv[:,0],uv[:,1],s=5,c='#ffef39',alpha=.8,linewidths=0)
  axs[1,0].set_xlim(0,W);axs[1,0].set_ylim(H,0)
  axs[1,0].set_title(f'B independent mask={bmask_index}; sealed visible foreground support n={len(points)} (partial, not amodal boundary)',fontsize=10)
  axs[1,1].imshow(rgb);overlay_mask(axs[1,1],amask,'#ffe600',.08)
  for name,color in COLORS.items():
   draw_box(axs[1,1],boxes[names.index(name)],camera,(W,H),color,name,1.7,'--' if name=='B_old_surface_equality' else '-')
  if alluv:
   u=np.concatenate(alluv);lo=np.maximum(u.min(0)-[50,50],[0,0]);hi=np.minimum(u.max(0)+[50,50],[W,H])
   if np.all(hi-lo>10):axs[1,1].set_xlim(lo[0],hi[0]);axs[1,1].set_ylim(hi[1],lo[1])
  axs[1,1].set_title('Same-view zoom; near-identical boxes may overlap — projection is not visual ground truth',fontsize=10)
  source_score=float(scores[q0]);qa=int(queries[names.index('A_full_consensus')]);ascore=float(scores[qa])
  title=f"{alias} | {selection['platform']} | {selection['selection_rule']} | source IoU(sealed)={selection['source_iou_sealed']:.6f} | textscore q0={source_score:.5g}, qA={ascore:.5g}"
  caption='PRIVATE literal caption: '+row['caption']
  status='Statuses: A='+selection['arm_status_sealed']['A_full_consensus']+'; Bnew='+selection['arm_status_sealed']['B_inequality_full']+'; Bold='+selection['arm_status_sealed']['B_old_surface_equality']
  fig.suptitle(title,fontsize=12,y=.995)
  fig.text(.015,.037,'\n'.join(textwrap.wrap(caption,175)),fontsize=8,va='bottom')
  fig.text(.015,.012,'\n'.join(textwrap.wrap(status+' | POST-SCORE descriptive selection; no new GT drawing/IoU/inference/optimization. Local media only.',200)),fontsize=7,va='bottom')
  fig.subplots_adjust(left=.01,right=.99,bottom=.11,top=.95,wspace=.015,hspace=.07)
  path=output/(f'{len(index)+1:02d}_{alias}.jpg');fig.savefig(path,format='jpg',dpi=140,pil_kwargs={'quality':94});plt.close(fig)
  index.append({**selection,'path':str(path.resolve()),'sha256':sha(path),'original_image_sha256':fr['image_sha256'],
       'A_selected_mask_index':amask_index,'B_selected_mask_index':bmask_index,'B_sealed_support_n':len(points),
       'source_text_score':source_score,'associated_original_query_text_score':ascore,
       'literal_caption_private':row['caption'],'no_GT_metadata_index_or_new_IoU_read':True})
 dump(output/'INDEX_PRIVATE.json',{'state':'rendered_R13_postscore_fixed_rule_descriptive_gallery_no_GT_boxes',
    'n':len(index),'plan_sha256':sha(planpath),'selection_duplicate_rules_not_replaced':plan['duplicate_rules_not_replaced'],
    'selection_missing_rules_not_replaced':plan['missing_rules_not_replaced'],'images_PRIVATE':index,
    'blind_or_semantic_identity_truth_claimed':False,'raw_media_public_export_permitted':False})
 dump(output/'SEALED.json',{'state':'sealed_R13_private_gallery','n_images':len(index),'plan_sha256':sha(planpath),
       'program_sha256':sha(Path(__file__)),'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
 print(json.dumps({'n':len(index),'seal_sha256':sha(output/'SEALED.json'),'jpg_paths':[r['path'] for r in index]},indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path)
 p.add_argument('--prepare',action='store_true');p.add_argument('--plan-sha256');a=p.parse_args()
 if a.prepare:
  if a.output is None:p.error('--prepare requires --output')
  make_plan(a.plan,a.output)
 else:
  if not a.plan_sha256:p.error('--plan-sha256 required')
  render(a.plan,a.plan_sha256)
