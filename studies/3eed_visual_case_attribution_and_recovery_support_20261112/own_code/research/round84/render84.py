"""Local PRIVATE scientific atlas: original media, projections, ranks and error maps."""
from core84 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.spatial import cKDTree
import itertools,textwrap,html,time
sys.path.insert(0,str(ROOT/'crossplatform/round7'))
from camera_projection import CameraSpec,read_camera_allowlist,project_model_points,raw_to_model_points
COL={'source':'#009ed2','frozen':'#d62728','raw':'#2ca02c','BT':'#8b49c8','oracle':'#efa600'}
SHORT={'source':'S','frozen':'F','raw':'R','BT':'B','oracle':'O'}
O=B84/'atlas_PRIVATE';P=B84/'capture33_attempt001';O.mkdir();(O/'pages').mkdir();summaries=[];regions=[];start=time.perf_counter()
def corners(b):return b[:3]+np.array(list(itertools.product([-1,1],repeat=3)))*b[3:]/2
EDGES=[(a,b) for a in range(8) for b in range(a+1,8) if bin(a^b).count('1')==1]
def bounds(uv,w,h):
 good=np.isfinite(uv).all(1)
 if not good.any():return None
 lo=np.maximum(uv[good].min(0),[0,0]);hi=np.minimum(uv[good].max(0),[w,h]);return None if (hi<=lo).any() else np.r_[lo,hi]
def cropbounds(rect,w,h,pad=.35):
 if rect is None:return np.array([0,0,w,h])
 center=(rect[:2]+rect[2:])/2;size=np.maximum((rect[2:]-rect[:2])*(1+pad*2),[100,100]);return np.r_[np.maximum(center-size/2,0),np.minimum(center+size/2,[w,h])]
def paint(ax,rgb,projection,chosen,crop=None,title=''):
 ax.imshow(rgb);groups={}
 for state,q in chosen.items():
  if state not in COL:continue
  uv=projection[q];groups.setdefault(q,[]).append(SHORT[state])
  for a,b in EDGES:
   if np.isfinite(uv[[a,b]]).all():ax.plot(uv[[a,b],0],uv[[a,b],1],color=COL[state],lw=1.2,alpha=.9,ls='--' if state in ['frozen','BT'] else '-')
 for q,labels in groups.items():
  rect=bounds(projection[q],rgb.shape[1],rgb.shape[0])
  if rect is not None:ax.text(rect[0],max(8,rect[1]-2),'q'+str(q)+' '+','.join(labels),fontsize=7,color='black',bbox={'facecolor':'white','alpha':.75,'pad':1})
 if crop is not None:ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1])
 ax.set_title(title,fontsize=10);ax.set_axis_off()
def bev(ax,cloud,boxes,chosen,zoom=False):
 ax.scatter(cloud[:,0],cloud[:,1],s=.2,c='#666666',alpha=.3,rasterized=True)
 for state,q in chosen.items():
  if state not in COL:continue
  b=boxes[q];lo=b[:2]-b[3:5]/2;ax.add_patch(Rectangle(lo,b[3],b[4],fill=False,edgecolor=COL[state],lw=1.5,ls='--' if state in ['frozen','BT'] else '-'));ax.text(b[0],b[1],SHORT[state],color=COL[state],fontsize=8)
 if zoom:
  bs=boxes[list(chosen.values())];lo=np.min(bs[:,:2]-bs[:,3:5]/2,0)-5;hi=np.max(bs[:,:2]+bs[:,3:5]/2,0)+5;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
 else:
  lo=np.quantile(cloud[:,:2],.005,0);hi=np.quantile(cloud[:,:2],.995,0);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
 ax.set_aspect('equal',adjustable='box');ax.set_title('Candidate BEV (yaw=0)' if zoom else 'Scene BEV (0.5–99.5% display extent)',fontsize=9);ax.set_xlabel('model x / m',fontsize=8);ax.set_ylabel('model y / m',fontsize=8)
def mask84(r,g,repeat):
 seed=int(hashlib.sha256(('R76complementarymask|'+r['id']+'|'+str(repeat)).encode()).hexdigest()[:16],16);order=np.random.default_rng(seed).permutation(np.flatnonzero(g['depth_valid']));n=len(order)//4;im=np.zeros(1536,bool);im[order[:n]]=True;pm=g['visible']&np.isin(g['cell'],order[n:2*n]);pm[g['query_anchor_indices'].astype(int)]=False;return pm,im
def savefig(fig,path):fig.savefig(path,dpi=115,pil_kwargs={'quality':88});plt.close(fig)
for ci,c in enumerate(read(H84/'experiment/CASES33_PRIVATE.json')):
 r=c['row'];sid=r['id'];alias=c['case'];a=load(P/'samples_PRIVATE'/(sid+'.npz'));rec=read(P/(alias+'_PRIVATE.json'));chosen=rec['chosen_PRIVATE'];primary={k:chosen[k] for k in COL};boxes=a['boxes'];xyz=a['xyz'];keep=a['selected_query_indices'];rgb=np.array(Image.open(r['imagepath']).convert('RGB'));h,w=rgb.shape[:2];fields=read_camera_allowlist(r['metapath']);cam=CameraSpec.from_camera_fields(r['platform'],fields,missing_distortion_policy='assume_zero_declared');projection=np.array([project_model_points(corners(b),cam,(w,h))['uv'] for b in boxes]);sp=project_model_points(xyz,cam,(w,h));raw=(np.load(r['pointpath'],allow_pickle=False) if Path(r['pointpath']).suffix=='.npy' else np.fromfile(r['pointpath'],np.float32).reshape(-1,4))[:,:3];sample=raw_to_model_points(raw[a['point_indices'].astype(int)],r['platform']);near=cKDTree(sample).query(xyz)[0];assert near.max()<1e-4;cloud=raw_to_model_points(raw[::max(1,len(raw)//40000)],r['platform']);orect=bounds(projection[chosen['oracle']],w,h);ocrop=cropbounds(orect,w,h);alluv=np.concatenate([projection[q] for q in primary.values()]);ucrop=cropbounds(bounds(alluv,w,h),w,h,.15)
 # Overview: no loss/attention interpretation is encoded in the colors.
 fig=plt.figure(figsize=(18,17));gs=gridspec.GridSpec(4,3,height_ratios=[1.35,1.0,.9,.5]);fig.subplots_adjust(top=.85,bottom=.035,hspace=.3,wspace=.2);fig.text(.025,.98,alias+' — saved predictions; GT oracle is diagnostic only',fontsize=16,va='top');fig.text(.025,.952,'\n'.join(textwrap.wrap(r['caption'],155)),fontsize=10,va='top');paint(fig.add_subplot(gs[0,:2]),rgb,projection,primary,title='Complete RGB: S=original, F=prepared, R=raw TTA, B=backtracking, O=A256 oracle');paint(fig.add_subplot(gs[0,2]),rgb,projection,primary,ocrop,'Oracle-candidate neighborhood (posthoc crop)');bev(fig.add_subplot(gs[1,0]),cloud,boxes,primary,False);bev(fig.add_subplot(gs[1,1]),cloud,boxes,primary,True);paint(fig.add_subplot(gs[1,2]),rgb,projection,primary,ucrop,'Union of competing candidates + context')
 ax=fig.add_subplot(gs[2,:2]);ax.axis('off');labels=['query','IoU']+['S rank/p','F rank/p','R rank/p','B rank/p','wrong R rank/p','wrong B rank/p'];tab=[]
 for q in keep:
  vals=[str(q),f'{a["candidate_ious"][q]:.4f}']
  for st in ['source']+STATES84:
   v=next(v for v in rec['ranks_PRIVATE'][st] if v['query_PRIVATE']==q);vals.append(f'{v["rank"]} / {v["probability"]:.4f}')
  tab.append(vals)
 table=ax.table(cellText=tab,colLabels=labels,loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.5);ax.set_title('Same candidates across states: probability normalized per state',fontsize=10)
 ax=fig.add_subplot(gs[2,2]);v=np.array([[box_iou84(boxes[x],boxes[y]) for y in keep] for x in keep]);ax.imshow(v,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(len(keep)),keep,fontsize=8);ax.set_yticks(range(len(keep)),keep,fontsize=8)
 for x,y in itertools.product(range(len(keep)),repeat=2):ax.text(y,x,f'{v[x,y]:.2f}',ha='center',va='center',fontsize=7,color='white' if v[x,y]>.6 else 'black')
 ax.set_title('Pairwise AABB IoU (not identity labels)',fontsize=9)
 ax=fig.add_subplot(gs[3,:]);ax.axis('off');hy=rec['hybrid_PRIVATE'];line='Head × representation diagnostic: '+ ' | '.join(k.replace('_',' ')+f': q{v["query_PRIVATE"]}, IoU={v["IoU"]:.4f}' for k,v in hy.items());ax.text(0,.9,'\n'.join(textwrap.wrap(line,155)),fontsize=9,va='top');ax.text(0,.25,'Hybrid swaps are posthoc interventions, may be off-manifold; they are not deployable TTA methods.\nBox overlap / oracle candidate support cannot by themselves certify object identity.',fontsize=9,va='top');savefig(fig,O/'pages'/(alias+'_overview.jpg'))
 # Show the same candidate in every state, rather than only each state's winner.
 for kind in ['point','RGB']:
  states=['source','frozen','raw','BT'];fig,axs=plt.subplots(4,len(keep),figsize=(max(14,len(keep)*3.8),12),squeeze=False);fig.subplots_adjust(top=.91,bottom=.04,hspace=.17,wspace=.03);fig.suptitle(alias+' — '+('last-decoder query→seed attention' if kind=='point' else 'query→seed→local RGB one-hop attention path')+'\nReading clues only: not object masks, not causal attribution',fontsize=13)
  vmax=max(float(a[('source_attention' if st=='source' else st+'_seed_attention')].max()) if kind=='point' else (0 if st=='source' else float(a[st+'_RGB_path_mass'].max())) for st in states);vmax=max(vmax,1e-6)
  for si,st in enumerate(states):
   for qi,q in enumerate(keep):
    ax=axs[si,qi];ax.imshow(rgb)
    if kind=='point':
     mass=a['source_attention' if st=='source' else st+'_seed_attention'][qi];valid=sp['in_image'];ax.scatter(sp['uv'][valid,0],sp['uv'][valid,1],c=mass[valid],cmap='magma',vmin=0,vmax=vmax,s=5,alpha=.8);subtitle=f'{st}, q{q}; attention sum={mass.sum():.3f}'
    elif st!='source':
     mass=a[st+'_RGB_path_mass'][qi].reshape(32,48);ax.imshow(np.ma.masked_where(mass<=0,mass),extent=[0,w,h,0],cmap='magma',vmin=0,vmax=vmax,alpha=.65,interpolation='nearest');subtitle=f'{st}, q{q}; assigned path mass={mass.sum():.3f}'
    else:subtitle='Original source: no RGB branch'
    rect=bounds(projection[q],w,h)
    if rect is not None:ax.add_patch(Rectangle(rect[:2],rect[2]-rect[0],rect[3]-rect[1],fill=False,edgecolor='cyan',lw=1))
    ax.set_title(subtitle,fontsize=8);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_axis_off()
  savefig(fig,O/'pages'/(alias+'_support_'+kind+'.jpg'))
 # Error deltas share scale across raw/BT and fitting/third-mask per signal.
 fig,axs=plt.subplots(4,3,figsize=(17,15));fig.subplots_adjust(top=.91,bottom=.04,hspace=.2,wspace=.1);fig.suptitle(alias+' — local weighted recovery-error changes vs frozen\nBLUE=error decreased; RED=increased; transparent=not supervised. Fixed scale per signal.\nOracle rectangle/box is a geometric proxy, not segmentation; third mask overlaps fitting targets.',fontsize=12);insideP=in_box84(xyz,boxes[chosen['oracle']]);gridxy=np.stack(np.meshgrid((np.arange(48)+.5)*w/48,(np.arange(32)+.5)*h/32),-1).reshape(-1,2);insideI=np.zeros(1536,bool) if orect is None else ((gridxy>=orect[:2])&(gridxy<=orect[2:])).all(1);g0=load(casepath(r) if sid in DEV_IDS else ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(sid+'.npz'));masks0=[mask84(r,g0,v) for v in [76001,76002,76003]]
 for col,sig in enumerate(['P','RGB','D']):
  data=[]
  for st,view in [('raw','fit'),('BT','fit'),('raw','third'),('BT','third')]:
   v=a[st+'_'+sig+'_maps']-a['frozen_'+sig+'_maps'];delta=v[:2].mean(0) if view=='fit' else v[2];data.append((st,view,delta))
  limit=max(max(float(abs(d).max()) for _,_,d in data),1e-10)
  for row,(st,view,delta) in enumerate(data):
   ax=axs[row,col];ax.imshow(rgb)
   if sig=='P':valid=sp['in_image']&((masks0[0][0]|masks0[1][0]) if view=='fit' else masks0[2][0]);im0=ax.scatter(sp['uv'][valid,0],sp['uv'][valid,1],c=delta[valid],cmap='bwr',vmin=-limit,vmax=limit,s=9,alpha=.9);inside=insideP
   else:
    mask=g0['depth_valid'] if sig=='D' else ((masks0[0][1]|masks0[1][1]) if view=='fit' else masks0[2][1]);im0=ax.imshow(np.ma.masked_where(~mask.reshape(32,48),delta.reshape(32,48)),extent=[0,w,h,0],cmap='bwr',vmin=-limit,vmax=limit,alpha=.8,interpolation='nearest');inside=insideI
   if orect is not None:ax.add_patch(Rectangle(orect[:2],orect[2]-orect[0],orect[3]-orect[1],fill=False,edgecolor='#ffb000',lw=1.4))
   ax.set_title(f'{st} / {view} / {sig}: total Δ={delta.sum():+.3g}',fontsize=9);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_axis_off();fig.colorbar(im0,ax=ax,shrink=.7,pad=.01);baseerr=a['frozen_'+sig+'_maps'][:2].mean(0) if view=='fit' else a['frozen_'+sig+'_maps'][2];regions.append({'case':alias,'state':st,'view':view,'signal':sig,'total_delta':float(delta.sum()),'oracle_proxy_delta':float(delta[inside].sum()),'outside_proxy_delta':float(delta[~inside].sum()),'absolute_delta_fraction_inside_proxy':float(abs(delta[inside]).sum()/max(abs(delta).sum(),1e-30)),'frozen_error_fraction_inside_proxy':float(baseerr[inside].sum()/max(baseerr.sum(),1e-30)),'proxy_has_no_instance_identity_certificate':True})
 savefig(fig,O/'pages'/(alias+'_loss.jpg'))
 geom={'case':alias,'image_size':[w,h],'seed_raw_coordinate_max_error':float(near.max()),'oracle_query_PRIVATE':chosen['oracle'],'oracle_candidate_IoU':float(a['candidate_ious'][chosen['oracle']]),'selected_queries_PRIVATE':chosen,'projected_visible_seed_count':int(sp['in_image'].sum()),'distinct_raw_points':len(raw),'figures':4};dump(O/(alias+'_GEOMETRY_PRIVATE.json'),geom);summaries.append({'case':alias,'platform':r['platform'],'oracle_candidate_IoU':geom['oracle_candidate_IoU'],'seed_raw_coordinate_max_error':geom['seed_raw_coordinate_max_error'],'projected_visible_seed_count':geom['projected_visible_seed_count'],'source_IoU':c['task']['IoU']['source'],'frozen_IoU':c['task']['IoU']['frozen_bi'],'raw_IoU':c['task']['IoU']['R80_raw_correct'],'BT_IoU':c['task']['IoU']['bt_correct']});print('ATLAS84',ci+1,alias,flush=True)
dump(H84/'CASE_PATH_ROWS.json',summaries);dump(H84/'LOSS_REGION_PROXY_ROWS.json',regions)
items=[]
for c in read(H84/'experiment/CASES33_PRIVATE.json'):
 alias=c['case'];items.append('<section id="'+alias+'"><h2>'+alias+'</h2><p>'+html.escape(c['row']['caption'])+'</p>'+''.join('<p><a href="pages/'+alias+'_'+kind+'.jpg"><img loading="lazy" style="max-width:100%" src="pages/'+alias+'_'+kind+'.jpg"></a></p>' for kind in ['overview','support_point','support_RGB','loss'])+'</section>')
(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R84 private 33-case atlas</title><style>body{font:16px sans-serif;max-width:1800px;margin:auto;background:#eee}section{background:white;padding:20px;margin:25px 0}nav{position:sticky;top:0;background:white;padding:10px}a{margin-right:10px}</style><h1>R84 PRIVATE — original media and captions, do not publish</h1><p>All31 R82 IoU changes + two explicit opposite-path cases. Oracle is offline diagnosis; no new algorithm.</p><nav>'+''.join('<a href="#'+c['case']+'">'+c['case']+'</a>' for c in read(H84/'experiment/CASES33_PRIVATE.json'))+'</nav>'+''.join(items));dump(O/'COUNTS.json',{'cases':33,'pages':132,'seconds':time.perf_counter()-start,'raw_media_originals_not_copied':True,'PRIVATE_NOT_FOR_GITHUB':True});seal(O,'R84_PRIVATE_ORIGINAL_MEDIA_33_CASE_ATLAS',n=33);print(read(O/'COUNTS.json'))
