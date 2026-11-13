"""PRIVATE: real frames/captions and fixed-box overlays for factorial and formal readouts."""
from pathlib import Path
import sys,json,numpy as np,itertools,html,textwrap
R=Path(__file__).resolve().parents[2];H=R/'research/round92';B=R/'crossplatform/round92';read=lambda p:json.loads(p.read_text());sys.path.insert(0,str(R/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
cases=read(H/'experiment/GOOD_BAD_CASES_PRIVATE.json');ys={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(R/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};O=B/'atlas_PRIVATE';O.mkdir(exist_ok=False);edges=[(a,b) for a in range(8) for b in range(a+1,8) if bin(a^b).count('1')==1];coverage=[];pages=[]
show=['R91_frozen','R91_bt_dynamic','R91_raw_dynamic','same_state_bt_residual_only','static_residual_formal_bt','same_state_raw_static_residual']
for c in cases:
 r=c['row'];t=c['task'];sid=r['id'];z=np.load(B/'prediction336_attempt001/samples_PRIVATE'/(sid+'.npz'));boxes=z['source_boxes'];q=t['queries'];comp=np.load(B/'prediction336_attempt001/components_PRIVATE'/(sid+'.npz'));rec=read(B/'prediction336_attempt001/records_PRIVATE'/(sid+'.json'));rgb=np.array(Image.open(r['imagepath']).convert('RGB'));h,w=rgb.shape[:2];cam=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');corn=boxes[:,None,:3]+np.array(list(itertools.product([-1,1],repeat=3)))[None]*boxes[:,None,3:]/2;uv=np.array([project_model_points(v,cam,(w,h))['uv'] for v in corn]);oracle=int(np.argmax(ys[sid]));fig,axs=plt.subplots(3,3,figsize=(17,14));fig.subplots_adjust(top=.87,hspace=.23);fig.suptitle(c['case']+' '+r['platform']+' / '+','.join(c['reasons']),fontsize=11);fig.text(.025,.95,'\n'.join(textwrap.wrap(r['caption'],150)),fontsize=9,va='top')
 def overlay(ax,query,col):
  for a,b in edges:
   if np.isfinite(uv[query,[a,b]]).all():ax.plot(uv[query,[a,b],0],uv[query,[a,b],1],c=col,lw=1)
 for ax,arm in zip(axs[:2].flat,show):
  ax.imshow(rgb);overlay(ax,oracle,'#00bbff');overlay(ax,q[arm],'#ff2200');ax.set_title(arm+f' IoU={t["IoU"][arm]:.4f}',fontsize=9);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
 ax=axs[2,0];ax.imshow(rgb);cs=[q[a] for a in ['R91_frozen','R91_bt_dynamic','static_residual_formal_bt']]
 for qi,col in zip(cs,['#0088ff','#ffaa00','#ff2200']):overlay(ax,qi,col)
 v=uv[cs+[oracle]].reshape(-1,2);v=v[np.isfinite(v).all(1)]
 if len(v):lo=np.maximum(v.min(0)-35,[0,0]);hi=np.minimum(v.max(0)+35,[w,h]);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
 ax.axis('off');ax.set_title('PRIVATE crop: blue=zero/orange=oldBT/red=newBT',fontsize=8)
 ax=axs[2,1];ax.axis('off');table=[]
 for qi in dict.fromkeys(cs+[q['same_state_raw_static_residual'],oracle]):
  vv=[str(qi),f'{ys[sid][qi]:.3f}']
  for a in ['R91_frozen','R91_bt_dynamic','same_state_bt_residual_only','static_residual_formal_bt','same_state_raw_static_residual']:
   p=z['scores'][z['arm_names'].tolist().index(a)];vv.append(str(int((p>p[qi]).sum())+1))
  table.append(vv)
 tab=ax.table(cellText=table,colLabels=['q','IoU','zero','oldBT','res-only','newBT','raw-hold'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(7);tab.scale(1,1.6)
 ax=axs[2,2];ax.axis('off');b0=comp['view0_base'];r0=comp['view0_residual'];key=rec['same_state_bt_static_residual']['feature_key'];bp=comp[key+'_base'];rp=comp[key+'_residual'];i=q['R91_frozen'];j=q['R91_bt_dynamic'];text='Fixed old-zero vs old-BT candidates\nbase margin0='+str(round(float(b0[i]-b0[j]),6))+'\nresidual margin0='+str(round(float(r0[i]-r0[j]),6))+'\nbase margin change='+str(round(float(bp[i]-bp[j]-b0[i]+b0[j]),6))+'\nresidual margin change='+str(round(float(rp[i]-rp[j]-r0[i]+r0[j]),6))+'\noldBT alpha='+str(read(R/'crossplatform/round91/prediction336_attempt001/records_PRIVATE'/(sid+'.json'))['R76_full_disc_physical_bt']['alpha'])+'\nnewBT alpha='+str(rec['static_residual_formal_bt']['alpha'])+'\nCyan box: offline candidate oracle only';ax.text(0,.9,text,va='top',fontsize=8)
 fig.savefig(O/(c['case']+'.jpg'),dpi=100);plt.close(fig);pages.append('<section><h2>'+c['case']+'</h2><p>'+html.escape(r['caption'])+'</p><img style="width:100%" src="'+c['case']+'.jpg"></section>');coverage.append({'case':c['case'],'platform':r['platform'],'full_scene_shown':True,'same_state_factorial_and_formal_readouts_shown':True,'actual_old_and_new_alpha_shown':True,'candidate_oracle_offline_only':True,'semantic_identity_requires_review':True})
(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R92 PRIVATE</title>'+''.join(pages));(H/'ATLAS_COVERAGE_ROWS.json').write_text(json.dumps(coverage,indent=2)+'\n');print({'pages':len(cases),'new_task_scores':False})
