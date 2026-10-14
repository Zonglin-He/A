"""Private all-window RGB contact sheets; public scalar-only chart."""
import json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
B=R/'artifacts/stvg_local_motion_regions_v1';O=R/'artifacts/stvg_short_event_tracklets_v1';M=R/'artifacts/stvg_matched_support_reference_v1'
def main():
 import numpy as np,matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
 read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 bar=read(B/'CPU_GENERATOR_BARRIER.json');assert bar['status']=='complete';rep=read(B/'ROOT_POSTSEAL_GEOMETRY.json');roster=read(M/'ROSTER_LOCK.json')['records'];cases={r['case_id']:load_source_case(R/r['source_case_path'])for r in roster};truth,_=_truths_after_seal(cases)
 out=B/'private_views';out.mkdir(exist_ok=True);products=[]
 for cid,c in cases.items():
  alias=('HC'if cid.startswith('hc2')else'V')+f'{int(cid[-3:])+1:02d}';z=read(B/'cases'/f'{cid}.json');rgb=np.load(O/'private_rgb'/f'{cid}.npz')['rgb_uint8'];assert len(rgb)==20
  fig,axs=plt.subplots(4,5,figsize=(16,10));colors=['#e4a13a','#15c4ed','#eb54dc','#dceb25','#6955f5']
  for wi,w in enumerate(z['windows']):
   for fi,ax in enumerate(axs[wi]):
    im=rgb[wi*5+fi];h,ww=im.shape[:2];ax.imshow(im)
    for cc in w['candidates']:
     fr=cc['frames'][fi]
     if not fr['valid']:continue
     x1,y1,x2,y2=np.array(fr['box_xyxy_norm'])*[ww,h,ww,h];col=colors[cc['candidate_id']-1];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=col,linewidth=1.4));ax.text(x1,y1,str(cc['candidate_id']),color='white',fontsize=8,bbox=dict(facecolor=col,alpha=.85,pad=1))
    fid=w['physical_frame_ids'][fi];gt=truth[cid]['truth'].get(fid)
    if gt is not None:
     x1,y1,x2,y2=gt;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='lime',linewidth=1.7,linestyle='--'))
    ax.set_title(f'W{wi+1} / offset {fi-2:+d} / '+('GT known'if gt is not None else'GT unknown'),fontsize=8);ax.axis('off')
  fig.suptitle(f'{alias}: fixed motion regions; ID1 Source; IDs2-5 local only; dashed green GT diagnostic',fontsize=12);fig.tight_layout();p=out/f'{alias}.png';assert not p.exists();fig.savefig(p,dpi=130);plt.close(fig);products.append(dict(path=str(p.relative_to(R)),sha256=sha(p),private=True,panels=20))
 fig,axs=plt.subplots(1,2,figsize=(10,4.5))
 for ax,support in zip(axs,['primary19','all_known23']):
  gs=[x for x in rep['groups']if x['support']==support];xx=np.arange(3)
  ax.bar(xx-.18,[x['old_mean_delta_pool_pp']for x in gs],.36,label='Old pool',color='#cf9140');ax.bar(xx+.18,[x['new_mean_delta_pool_pp']for x in gs],.36,label='Motion-region pool',color='#357bb0');ax.set_xticks(xx,['All','HC','VidSTG']);ax.set_ylabel('Mean candidate-oracle advantage over Source (pp)');ax.set_title(support);ax.legend();ax.axhline(0,color='black',lw=.6)
 fig.suptitle('Center geometry only: no new R1 calls or OPD; 9 windows have unknown GT');fig.tight_layout();p=B/'figures/local_motion_regions.png';p.parent.mkdir(exist_ok=True);assert not p.exists();fig.savefig(p,dpi=170);plt.close(fig);products.append(dict(path=str(p.relative_to(R)),sha256=sha(p),private=False,panels=2))
 (B/'FIGURE_EXPORT.json').write_text(json.dumps(dict(status='exported_pending_actual_pixel_view',scope='8 private sheets all160frames and public scalar chart',files=products,root_geometry_sha256=sha(B/'ROOT_POSTSEAL_GEOMETRY.json'),GT_read=True,public_media=False),indent=2)+'\n')
 print(json.dumps(dict(status='exported',private_sheets=8,frames=160,public_charts=1)))
if __name__=='__main__':main()
