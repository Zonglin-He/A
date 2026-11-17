"""Private postscore RGB-cache/BEV case pages; no method revision or original media read."""
from pathlib import Path
import json,numpy as np,hashlib,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
R=Path(__file__).resolve().parents[2];H=R/'research/round127';B=R/'crossplatform/round127';rd=lambda p:json.loads(Path(p).read_text())
def corners(box):
 c=np.asarray(box[:2]);s=np.maximum(np.asarray(box[3:5]),1e-6);yaw=box[6] if len(box)>6 else 0.;p=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*s/2;cs,sn=np.cos(yaw),np.sin(yaw);return p@np.array([[cs,sn],[-sn,cs]])+c
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');er={(e['cohort'],e['source'],e['row']['id']):e for e in reg};rows=[r for p in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)];chosen=[]
 for co in ['DEV','EXP']:
  for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
   for kind,sgn in [('good',1),('failure',-1)]:
    rr=[r for r in rows if r['cohort']==co and r['source']==src and r['target']==tgt and not r['protected_alias_fallback'] and sgn*(r['arms']['fab_full_F']['IoU']-r['arms']['zero_F']['IoU'])>1e-12]
    if rr:chosen.append((kind,max(rr,key=lambda r:sgn*(r['arms']['fab_full_F']['IoU']-r['arms']['zero_F']['IoU']))))
 D=B/'private_atlas_attempt001';D.mkdir(exist_ok=False);index=[];colors=['tab:gray','tab:blue','tab:orange','tab:green','tab:purple'];methods=['original_source','zero_F','direct_full_F','fab_full_F','ab_native_F']
 for ix,(kind,row) in enumerate(chosen):
  e=er[(row['cohort'],row['source'],row['id_PRIVATE'])];assert not e['protected_alias_fallback'];z=np.load(R/e['prefix']);key=hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest();g=np.load(R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(row['id_PRIVATE']+'_'+key[:20]+'.npz'));xyz=z['xyz'][0];sample=np.load(B/'formal'/row['cohort']/(row['source']+'_to_'+row['target'])/'samples_PRIVATE'/(row['id_PRIVATE']+'.npz'));boxes=sample['source_boxes'];query={str(k):int(q) for k,q in zip(sample['arm_names'],sample['queries'])};master=rd(R/f"research/round96/experiment/OFFICIAL_{row['target'].upper()}_VAL_ORDER_PRIVATE.json");rs=rd(R/f"research/round{99 if row['cohort']=='DEV' else 109}/input/TARGET_{row['target'].upper()}{128 if row['cohort']=='DEV' else 256}_PRIVATE.json");item=next(r for r in rs if r['id']==row['id_PRIVATE']);gt=np.array(master[item['anno_index']]['source_GT7_float32']);fig,axes=plt.subplots(1,3,figsize=(18,7));axes[0].imshow(g['image']);axes[0].set_title('Cached RGB + binned projection of in-box seeds\nNot attention or an instance mask');axes[0].axis('off');points=[]
  for arm,color in zip(methods,colors):
   b=boxes[query[arm]];points.append(corners(b));inside=np.all(np.abs(xyz-b[:3])<=np.maximum(b[3:6],1e-6)/2,axis=1);vis=np.asarray(g['visible']).astype(bool);cell=np.asarray(g['cell']).astype(int);use=inside&vis&(cell>=0);axes[0].scatter((cell[use]%48+.5)*8,(cell[use]//48+.5)*8,s=8,c=color,label=arm,alpha=.7)
   for ax in axes[1:]:ax.add_patch(Polygon(corners(b),fill=False,edgecolor=color,linewidth=2,label=f'{arm}: {row["arms"][arm]["IoU"]:.3f}'))
  for ax in axes[1:]:ax.scatter(xyz[:,0],xyz[:,1],s=1,c='lightgray');ax.add_patch(Polygon(corners(gt),fill=False,edgecolor='red',linestyle='--',linewidth=2,label='GT, postscore only'));ax.set_aspect('equal',adjustable='box');ax.set_xlabel('Canonical x');ax.set_ylabel('Canonical y')
  axes[1].set_title('Full seed BEV');axes[2].set_title('Candidate competition, local BEV');pts=np.concatenate(points+[corners(gt)]);axes[2].set_xlim(pts[:,0].min()-3,pts[:,0].max()+3);axes[2].set_ylim(pts[:,1].min()-3,pts[:,1].max()+3);axes[2].legend(fontsize=8);delta=row['arms']['fab_full_F']['IoU']-row['arms']['zero_F']['IoU'];caption=e['row']['caption'];fig.suptitle(f'CASE{ix+1:03d} {kind} {row["cohort"]} {row["source"]}->{row["target"]}; deltaIoU={delta:+.4f}\n'+textwrap.fill(caption,160),fontsize=11);fig.tight_layout(rect=(0,.03,1,.87));name=f'CASE{ix+1:03d}.png';fig.savefig(D/name,dpi=130);plt.close(fig);index.append({'case':name,'kind':kind,'cohort':row['cohort'],'source':row['source'],'target':row['target'],'id_PRIVATE':row['id_PRIVATE'],'delta_IoU':delta,'identity':'UNKNOWN_pending_private_visual_review','media_source':'already-used fixed cached RGB and seed geometry only; no original media decoded'})
 (D/'INDEX_PRIVATE.json').write_text(json.dumps(index,indent=2)+'\n');
 (D/'INDEX.html').write_text('<!doctype html><meta charset="utf-8"><title>R127 private descriptive cases</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;padding:20px;background:#f5f5f5}section{background:white;padding:20px;margin:24px 0}img{width:100%}</style><h1>R127 private postscore cases</h1><p>Cached RGB and seed geometry only. Red dashed boxes are GT used after prediction sealing. Geometric in-box seeds are neither attention nor certified instance masks. These cases never select the method.</p>'+''.join('<section><h2>'+x['case']+' '+x['kind']+' '+x['cohort']+' '+x['source']+' to '+x['target']+' delta IoU '+str(round(x['delta_IoU'],5))+'</h2><a href="'+x['case']+'"><img src="'+x['case']+'"></a></section>' for x in index))
 (H/'PRIVATE_ATLAS_COUNTS.json').write_text(json.dumps({'n':len(index),'good':sum(x['kind']=='good' for x in index),'failure':sum(x['kind']=='failure' for x in index),'postscore_only':True,'used_for_method_selection':False,'original_media_access':False,'FINAL_access':False},indent=2)+'\n');print('PRIVATE_ATLAS127',len(index),flush=True)
if __name__=='__main__':main()
