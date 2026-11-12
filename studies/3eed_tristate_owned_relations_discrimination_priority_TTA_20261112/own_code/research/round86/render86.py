"""Private original-image case pages; never included in public export."""
from core86 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import itertools,textwrap,html
sys.path.insert(0,str(ROOT/'crossplatform/round7'))
from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
P=B86/'prediction366_attempt001';O=B86/'atlas_PRIVATE_attempt002';O.mkdir(exist_ok=False);tasks={x['id_PRIVATE']:x for p in ['dev336','supp30'] for x in read(H86/('score_'+p+'_attempt001/ALL_TASK_ROWS_PRIVATE.json'))};cases=read(ROOT/'research/round84/experiment/CASES33_PRIVATE.json');ys={}
for f in ['research/round55/score_dev336_attempt001','research/round69/score_all2304_attempt001']:ys.update({x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(ROOT/f/'ALL_CANDIDATE_IOUS_PRIVATE.json')})
edges=[(a,b) for a in range(8) for b in range(a+1,8) if bin(a^b).count('1')==1];items=[];outrows=[]
for ci in [21,4,20]:
 c=cases[ci];r=c['row'];sid=r['id'];alias='CASE'+str(ci+1).zfill(3);x=tasks[sid];a=load(P/'samples_PRIVATE'/(sid+'.npz'));ev=load(B86/'evidence366_attempt002/samples_PRIVATE'/(sid+'.npz'));meta=read(B86/'evidence366_attempt002/cases_PRIVATE'/(sid+'.json'));old=load(B85/'prediction366_attempt002/samples_PRIVATE'/(sid+'.npz'));rgb=np.array(Image.open(r['imagepath']).convert('RGB'));h,w=rgb.shape[:2];cam=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');boxes=a['source_boxes'];corners=boxes[:,:3,None]+np.array(list(itertools.product([-1,1],repeat=3))).T[None,:,:]*boxes[:,3:,None]/2;uv=np.array([project_model_points(p.T,cam,(w,h))['uv'] for p in corners]);oq=int(np.argmax(ys[sid]));arms=['source','frozen_bi','R85_full1','F_direct_guard','F_joint1_raw','F_joint1_guard'];qs={arm:x['queries'][arm] for arm in arms};qs['R85_direct']=int(old['queries'][old['arm_names'].tolist().index('direct_constraints')]);qs['oracle']=oq
 union=np.concatenate([uv[q] for q in qs.values()]);valid=union[np.isfinite(union).all(1)];lo=np.maximum(valid.min(0)-30,[0,0]);hi=np.minimum(valid.max(0)+30,[w,h]);fig,axs=plt.subplots(4,2,figsize=(16,16));fig.subplots_adjust(top=.91,hspace=.25,bottom=.025);fig.suptitle(alias+' — PRIVATE original media; oracle offline only',fontsize=15);fig.text(.02,.96,'\n'.join(textwrap.wrap(r['caption'],150)),fontsize=10,va='top')
 for ax,(title,arm) in zip(axs[:3].flat,[(k,k) for k in arms]):
  ax.imshow(rgb);q=qs[arm]
  for qq,col in [(oq,'#00aaff'),(q,'#ff3300')]:
   for aa,bb in edges:
    if np.isfinite(uv[qq,[aa,bb]]).all():ax.plot(uv[qq,[aa,bb],0],uv[qq,[aa,bb],1],color=col,lw=1.2)
  ax.set_title(f'{title}: q{q} IoU={x["IoU"][arm]:.4f}; cyan=oracle',fontsize=9);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off')
 ax=axs[3,0];ax.imshow(rgb);ax.axis('off');ax.set_title('Complete image / all selected candidate support',fontsize=9)
 z0=prefix(r) if sid in DEV_IDS else {'xyz':load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(sid+'.npz'),keys=['A_xyz'])['A_xyz']};proj=project_model_points(z0['xyz'][0],cam,(w,h));keep=np.array(proj['in_image']);inside=(abs(z0['xyz'][0]-boxes[qs['F_joint1_guard'],:3])<=boxes[qs['F_joint1_guard'],3:]/2).all(-1);sel=keep&inside;ax.scatter(proj['uv'][sel,0],proj['uv'][sel,1],s=4,c='orange',alpha=.7)
 ax=axs[3,1];ax.axis('off');tab=[];uqs=list(dict.fromkeys(qs.values()))
 for q in uqs:
  row=[str(q),str(ev['labels'][q]),f'{ys[sid][q]:.3f}']
  for arm in ['frozen_bi','F_direct_guard','F_candidate1_guard','F_joint1_guard']:
   v=a['scores'][ARMS86.index(arm)];row.append(f'{int((v>v[q]).sum())+1}/{v[q]:.3g}')
  group=int(ev['labels'][q]);row.append('; '.join(f'e{ki}:{s[group]:.2g}/{n[group]:.2g}' for ki,(k,s,n) in enumerate(zip(meta['factor_kinds'],ev['support'],ev['counter']))) or 'UNKNOWN');tab.append(row)
 table=ax.table(cellText=tab,colLabels=['q','group','IoU','F rank/p','direct rank/p','candidate rank/p','joint rank/p','support/counter'],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(6);table.scale(1,1.65);ax.set_title('Same candidates; ineligible guard scores=-1; no identity certificate',fontsize=8)
 fig.savefig(O/(alias+'.jpg'),dpi=115);plt.close(fig);record={'case':alias,'row_PRIVATE':sid,'caption_PRIVATE':r['caption'],'graph_PRIVATE':meta['graph'],'queries_PRIVATE':qs,'IoU':x['IoU'],'scores_and_factors_PRIVATE':tab,'pixel_support_not_identity':True};dump(O/(alias+'_PRIVATE.json'),record);items.append('<h2>'+alias+'</h2><p>'+html.escape(r['caption'])+'</p><img style="width:100%" src="'+alias+'.jpg">');outrows.append({'case':alias,'factor_count':len(meta['factor_kinds']),'new_joint_same_candidate_as_source':qs['F_joint1_guard']==qs['source'],'new_joint_same_candidate_as_frozen':qs['F_joint1_guard']==qs['frozen_bi'],'new_joint_same_candidate_as_direct':qs['F_joint1_guard']==qs['F_direct_guard'],'new_instance_identity_unknown':True})
(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>R86 PRIVATE case review</h1>'+''.join(items));dump(H86/'THREE_CASE_REVIEW_ROWS.json',outrows);print({'private_pages':3,'output':str(O)})
