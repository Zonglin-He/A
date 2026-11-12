"""Private atlas selected from global positive/negative tails, not handpicked fixes."""
from structural_core87 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import itertools,textwrap,html
sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
O=B87/'structural_atlas_PRIVATE';O.mkdir(exist_ok=False);P=B87/'structural_prediction336_attempt001';ys={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};edges=[(a,b) for a in range(8) for b in range(a+1,8) if bin(a^b).count('1')==1];items=[];geom=[]
for c in read(H87/'experiment/STRUCTURAL_FOLLOWUP_CASES_PRIVATE.json'):
 r=c['row'];sid=r['id'];alias=c['case'];t=c['task'];a=load(P/'samples_PRIVATE'/(sid+'.npz'));ec=load(B87/'structural_evidence336_attempt002/samples_PRIVATE'/(sid+'.npz'));ev,graph,es=prepared87(r);rgb=np.array(Image.open(r['imagepath']).convert('RGB'));h,w=rgb.shape[:2];cam=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');boxes=a['source_boxes'];corn=boxes[:,None,:3]+np.array(list(itertools.product([-1,1],repeat=3)))[None]*boxes[:,None,3:]/2;uv=np.array([project_model_points(q,cam,(w,h))['uv'] for q in corn]);oq=int(np.argmax(ys[sid]));arms=['source','frozen_bi','full_direct','full_tta1','wrong_tta1'];qs=[t['queries'][arm] for arm in arms];vv=np.concatenate([uv[q] for q in qs+[oq]]);valid=vv[np.isfinite(vv).all(1)];lo=np.maximum(valid.min(0)-40,[0,0]) if len(valid) else np.array([0,0]);hi=np.minimum(valid.max(0)+40,[w,h]) if len(valid) else np.array([w,h]);fig,axs=plt.subplots(4,2,figsize=(16,15));fig.subplots_adjust(top=.91,hspace=.3,bottom=.025);fig.suptitle(alias+' / '+r['platform']+' / '+';'.join(c['reasons']),fontsize=13);fig.text(.02,.965,'\n'.join(textwrap.wrap(r['caption'],145)),fontsize=9,va='top')
 for ax,arm in zip(axs[:3].flat,arms):
  ax.imshow(rgb);q=t['queries'][arm]
  for qq,col in [(oq,'#00bbff'),(q,'#ff3300')]:
   for aa,bb in edges:
    if np.isfinite(uv[qq,[aa,bb]]).all():ax.plot(uv[qq,[aa,bb],0],uv[qq,[aa,bb],1],c=col,lw=1.1)
  ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off');ax.set_title(arm+f' q{q} IoU={t["IoU"][arm]:.4f}; cyan=offline oracle',fontsize=9)
 ax=axs[2,1];ax.imshow(rgb);ax.axis('off');ax.set_title('Full scene / uncertain reference hypotheses',fontsize=9);belief=ec['full_beliefs']
 for i,entity in enumerate(graph['entities'][1:],1):
  q=int(belief[i,:256].argmax());v=uv[q]
  if np.isfinite(v).all():ax.text(v[:,0].mean(),v[:,1].mean(),entity['role']+f' null={belief[i,-1]:.2f}',color='blue',fontsize=7,bbox={'facecolor':'white','alpha':.8,'pad':1})
 ax=axs[3,0];ax.axis('off');table=[]
 for mode in MODES87+['wrong']:
  table.append([mode,f'{t["IoU"][mode+"_direct"]:.4f}',f'{t["IoU"][mode+"_tta1"]:.4f}',str(t['queries'][mode+'_tta1'])])
 tab=ax.table(cellText=table,colLabels=['component','direct IoU','TTA1 IoU','selected q'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.3)
 ax=axs[3,1];ax.axis('off');table=[];selected=list(dict.fromkeys(qs+[oq]))
 for q in selected:
  row=[str(q),f'{ys[sid][q]:.3f}']
  for arm in ['frozen_bi','full_direct','full_tta1','wrong_tta1']:
   v=a['scores'][ARMS87.index(arm)];row.append(f'{int((v>v[q]).sum())+1}/{v[q]:.3g}')
  table.append(row)
 tab=ax.table(cellText=table,colLabels=['q','IoU','frozen rank/p','direct rank/p','TTA rank/p','wrong rank/p'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(7);tab.scale(1,1.4)
 fig.savefig(O/(alias+'.jpg'),dpi=105);plt.close(fig);dump(O/(alias+'_PRIVATE.json'),{'row':r,'graph':graph,'selected_queries':dict(zip(arms,qs)),'reference_posterior':belief.tolist(),'score_table':table});items.append('<section><h2>'+alias+'</h2><p>'+html.escape(r['caption'])+'</p><p>'+html.escape(';'.join(c['reasons']))+'</p><img style="width:100%" src="'+alias+'.jpg"></section>');geom.append({'case':alias,'platform':r['platform'],'selected_candidates':len(selected),'visible_reference_hypotheses':len(graph['entities'])-1,'identity_unknown_unless_independently_reviewed':True})
(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R87 private global good/bad atlas</title><h1>R87 PRIVATE / original images and full captions</h1>'+''.join(items));dump(H87/'STRUCTURAL_ATLAS_COVERAGE_ROWS.json',geom);print({'cases':len(items),'PRIVATE':True})
cases=read(H87/'experiment/STRUCTURAL_FOLLOWUP_CASES_PRIVATE.json');fig,axs=plt.subplots(int(np.ceil(len(cases)/3)),3,figsize=(16,max(4,int(np.ceil(len(cases)/3))*3.7)),squeeze=False)
for ax,c in zip(axs.flat,cases):
 r=c['row'];t=c['task'];a=load(P/'samples_PRIVATE'/(r['id']+'.npz'));rgb=np.array(Image.open(r['imagepath']).convert('RGB'));h,w=rgb.shape[:2];cam=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');ax.imshow(rgb);alluv=[]
 for arm,col in [('frozen_bi','#0088ff'),('full_direct','#ffaa00'),('full_tta1','#ff2222')]:
  b=a['source_boxes'][t['queries'][arm]];corn=b[:3]+np.array(list(itertools.product([-1,1],repeat=3)))*b[3:]/2;v=project_model_points(corn,cam,(w,h))['uv'];alluv.append(v)
  for aa,bb in edges:
   if np.isfinite(v[[aa,bb]]).all():ax.plot(v[[aa,bb],0],v[[aa,bb],1],c=col,lw=1)
 v=np.concatenate(alluv);v=v[np.isfinite(v).all(1)]
 if len(v):lo=np.maximum(v.min(0)-60,[0,0]);hi=np.minimum(v.max(0)+60,[w,h]);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
 ax.set_title(c['case']+' '+r['platform']+f' F/D/T={t["IoU"]["frozen_bi"]:.3f}/{t["IoU"]["full_direct"]:.3f}/{t["IoU"]["full_tta1"]:.3f}\n'+r['caption'][:85],fontsize=7);ax.axis('off')
for ax in axs.flat[len(cases):]:ax.axis('off')
fig.suptitle('R87 structural — fixed original selected tails — blue=frozen / orange=direct / red=TTA; not identity labels',fontsize=11);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(O/'OVERVIEW.jpg',dpi=130);plt.close(fig)
