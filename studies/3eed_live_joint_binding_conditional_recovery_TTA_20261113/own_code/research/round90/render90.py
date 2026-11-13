"""Private full-context case pages: decisions, global/local priors, separate hypotheses."""
from target90 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import itertools,textwrap,html
sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
O=B90/'atlas_PRIVATE';O.mkdir(exist_ok=False);P=B90/'prediction336_attempt001';ys={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};edges=[(a,b) for a in range(8) for b in range(a+1,8) if bin(a^b).count('1')==1];cases=read(H90/'experiment/GOOD_BAD_CASES_PRIVATE.json');rows=read(H90/'experiment/DEV336_PRIVATE.json');tasks=read(H90/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');cases+=read(H90/'experiment/FIXED_R89_CASES_PRIVATE.json')
items=[];coverage=[]
for c in cases:
 r=c['row'];sid=r['id'];t=c['task'];a=load(P/'samples_PRIVATE'/(sid+'.npz'));ec=load(B89/'target_evidence336_attempt001/samples_PRIVATE'/(sid+'.npz'));info=read(B89/'target_evidence336_attempt001/records_PRIVATE'/(sid+'.json'));z0=prefix(r);g=load(casepath(r));rgb=np.array(Image.open(r['imagepath']).convert('RGB'));h,w=rgb.shape[:2];cam=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');boxes=a['source_boxes'];corn=boxes[:,None,:3]+np.array(list(itertools.product([-1,1],repeat=3)))[None]*boxes[:,None,3:]/2;uv=np.array([project_model_points(q,cam,(w,h))['uv'] for q in corn]);oracle=int(np.argmax(ys[sid]));fig,axs=plt.subplots(4,3,figsize=(18,18));fig.subplots_adjust(top=.87,hspace=.23);fig.suptitle(c['case']+' '+r['platform']+' / '+','.join(c['reasons']),fontsize=13);fig.text(.025,.955,'\n'.join(textwrap.wrap(r['caption'],160)),fontsize=9,va='top')
 for ax,arm in zip(axs[:2].flat,['source','frozen_bi','candidate_joint_frozen','candidate_joint_tta1','candidate_joint_feature3','wrong_space_tta1']):
  ax.imshow(rgb)
  for q,col in [(oracle,'#00bbff'),(t['queries'][arm],'#ff2200')]:
   for aa,bb in edges:
    if np.isfinite(uv[q,[aa,bb]]).all():ax.plot(uv[q,[aa,bb],0],uv[q,[aa,bb],1],c=col,lw=1)
  ax.set_title(arm+f' IoU={t["IoU"][arm]:.4f} / cyan=offline oracle',fontsize=10);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
 valid=g['visible'];xx=g['uv_grid'][:,0]/48*w;yy=g['uv_grid'][:,1]/32*h
 replay=load(B90/'case_replay_PRIVATE'/(c['case']+'.npz'))
 for ax,kind in zip(axs[2,:2],['global','local']):
  ax.imshow(rgb);tm=ec['token_roles'];att=replay['candidate_joint_tta1_native_attention'][tm[0]].mean(0) if len(tm) and tm[0].any() else np.zeros(1024);winner=t['queries']['candidate_joint_tta1'];ww=replay['candidate_joint_tta1_subject_weights'];field=att if kind=='global' else (ww[winner,0]*ec['own_support'][winner]+ww[winner,1:]@ec['supports'][0] if len(tm) else np.zeros(1024));sc=field/(field.max()+1e-12);ax.scatter(xx[valid],yy[valid],c=sc[valid],s=9,cmap='magma',vmin=0,vmax=1,alpha=.7);ax.set_title('Updated full-context subject attention' if kind=='global' else 'Updated selected-candidate support weights (not instance mask)',fontsize=9);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
 ax=axs[2,2];xyz=z0['xyz'][0];ax.scatter(xyz[:,0],xyz[:,1],s=2,c='lightgray')
 for e,entity in enumerate(info['graph']['entities']):
  for j,q in enumerate(ec['hypotheses_PRIVATE'][e]):
   if q<0:continue
   m=ec['supports'][e,j].astype(bool);ax.scatter(xyz[m,0],xyz[m,1],s=4,alpha=.35);mu=ec['support_centers_PRIVATE'][e,j];ax.text(mu[0],mu[1],entity['role']+str(j),fontsize=7)
 ax.set_title('Separate role hypotheses / BEV support; no averaged object',fontsize=8);ax.set_aspect('equal',adjustable='datalim')
 ax=axs[3,0];ax.imshow(rgb);chosen=[t['queries'][arm] for arm in ['frozen_bi','candidate_joint_frozen','candidate_joint_tta1']];vv=np.concatenate([uv[q] for q in chosen+[oracle]]);validuv=vv[np.isfinite(vv).all(1)]
 for q,col in zip(chosen,['#0088ff','#ffaa00','#ff2200']):
  for aa,bb in edges:
   if np.isfinite(uv[q,[aa,bb]]).all():ax.plot(uv[q,[aa,bb],0],uv[q,[aa,bb],1],c=col,lw=1.2)
 if len(validuv):lo=np.maximum(validuv.min(0)-35,[0,0]);hi=np.minimum(validuv.max(0)+35,[w,h]);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
 ax.axis('off');ax.set_title('Postseal diagnostic crop: blue=frozen/orange=direct/red=TTA',fontsize=8)
 ax=axs[3,1];ax.axis('off');table=[]
 for q in dict.fromkeys(chosen+[oracle]):
  row=[str(q),f'{ys[sid][q]:.3f}']
  for arm in ['frozen_bi','candidate_joint_frozen','candidate_joint_tta1','candidate_joint_feature3']:
   prob=a['scores'][a['arm_names'].tolist().index(arm)];row.append(str(int((prob>prob[q]).sum())+1))
  table.append(row)
 tab=ax.table(cellText=table,colLabels=['q','IoU','frozen rank','direct rank','TTA rank','3read rank'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(7);tab.scale(1,1.5)
 ax=axs[3,2];ax.axis('off');table=[[arm,f'{t["IoU"][arm]:.4f}'] for arm in ['candidate_joint_tta1','candidate_joint_tta3','query_joint_tta1','continue_joint_tta1','candidate_headonly_tta1','no_ref_tta1','candidate_joint_bypass1','candidate_joint_physical1']];tab=ax.table(cellText=table,colLabels=['component','IoU'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.3)
 fig.savefig(O/(c['case']+'.jpg'),dpi=95);plt.close(fig);items.append('<section><h2>'+c['case']+'</h2><p>'+html.escape(r['caption'])+'</p><img style="width:100%" src="'+c['case']+'.jpg"></section>');coverage.append({'case':c['case'],'platform':r['platform'],'full_scene_shown':True,'global_and_local_support_shown':True,'separate_role_hypotheses_shown':True,'identity_requires_independent_review':True})
(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R90 PRIVATE</title>'+''.join(items));dump(H90/'ATLAS_COVERAGE_ROWS.json',coverage);print({'pages':len(cases)})
