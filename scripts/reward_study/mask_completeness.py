"""Prespecified CPU-only diagnostics; cannot read GT before complete native-mask seal.

No inference, adaptation, temperature search, or reward selection is implemented.
Use --run-after-seal only after root has confirmed all 128 receipts are sealed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,json,hashlib,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.ndimage import label as connected_components
torch.set_num_threads(1)
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();TRIAL=ROOT/'outputs/layout_reward_dev';OUT=TRIAL/'research';MASKS=OUT/'native_masks'
SCALES=(.5,.75,1.,1.25,1.5,2.)
DUPLICATE_IOU=.9
SCORE_KEYS=('mask_mean','mask_coverage','pixel_Dice','projection_Dice','pixel_Dice_hard','projection_Dice_hard')
RANK_KEYS=SCORE_KEYS+('geometry_reference',)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def arr(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def summary(x):
 z=np.asarray(x,dtype=float);z=z[np.isfinite(z)]
 return dict(n=len(z),mean=float(z.mean()),median=float(np.median(z)),p10=float(np.quantile(z,.1)),p90=float(np.quantile(z,.9))) if len(z) else dict(n=0)
def corners(b):return np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]
def cxcywh(x):return np.r_[(x[:2]+x[2:])/2,x[2:]-x[:2]]
def geometry_reward(b,t):
 p=corners(b);q=corners(t);inter=np.maximum(np.minimum(p[2:],q[2:])-np.maximum(p[:2],q[:2]),0).prod();union=max(np.maximum(p[2:]-p[:2],0).prod()+np.maximum(q[2:]-q[:2],0).prod()-inter,1e-7);enclose=max(np.maximum(np.maximum(p[2:],q[2:])-np.minimum(p[:2],q[:2]),0).prod(),1e-7);giou=inter/union-(enclose-union)/enclose
 return float(-(5*np.abs(b-t).sum()+2*(1-giou)))
def biou(b,g):
 b=np.maximum(corners(np.asarray(b)),0);g=np.maximum(np.asarray(g),0)
 area=np.maximum(b[2:]-b[:2],0).prod();ag=np.maximum(g[2:]-g[:2],0).prod();inter=np.maximum(np.minimum(b[2:],g[2:])-np.maximum(b[:2],g[:2]),0).prod()
 return float(inter/max(area+ag-inter,1e-15))
def integrate(S,x,y):
 h,w=np.array(S.shape)-1;x=np.clip(x,0,w);y=np.clip(y,0,h);ix=int(np.floor(x));iy=int(np.floor(y));jx=min(ix+1,w);jy=min(iy+1,h);fx=x-ix;fy=y-iy
 return (1-fy)*((1-fx)*S[iy,ix]+fx*S[iy,jx])+fy*((1-fx)*S[jy,ix]+fx*S[jy,jx])
def integrate1(S,x):
 n=len(S)-1;x=np.clip(x,0,n);i=int(np.floor(x));j=min(i+1,n);return (1-(x-i))*S[i]+(x-i)*S[j]
class MaskScores:
 def __init__(self,P):
  self.P=np.asarray(P,dtype=np.float64);self.h,self.w=self.P.shape;self.mass=float(self.P.sum());self.S=np.pad(self.P.cumsum(0).cumsum(1),((1,0),(1,0)));self.px=self.P.max(0);self.py=self.P.max(1);self.pxs=np.r_[0,self.px.cumsum()];self.pys=np.r_[0,self.py.cumsum()]
 def score(self,b):
  x0,y0,x1,y1=corners(b)*np.array([self.w,self.h,self.w,self.h]);x0,x1=np.clip([x0,x1],0,self.w);y0,y1=np.clip([y0,y1],0,self.h);area=max(x1-x0,0)*max(y1-y0,0)
  mass=integrate(self.S,x1,y1)-integrate(self.S,x0,y1)-integrate(self.S,x1,y0)+integrate(self.S,x0,y0)
  mx=integrate1(self.pxs,x1)-integrate1(self.pxs,x0);my=integrate1(self.pys,y1)-integrate1(self.pys,y0)
  return dict(mask_mean=mass/max(area,1e-15),mask_coverage=mass/max(self.mass,1e-15),pixel_Dice=2*mass/max(area+self.mass,1e-15),projection_Dice=.5*(2*mx/max((x1-x0)+self.px.sum(),1e-15)+2*my/max((y1-y0)+self.py.sum(),1e-15)))

def validate_before_GT():
 assert (MASKS/'sealed.json').exists(),'Do not read GT: native masks not sealed'
 seal=read(MASKS/'sealed.json');assert seal['queries']==128 and seal['GT_read'] is False
 old=read(TRIAL/'scope_hc2/sealed.json');records=[json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l and json.loads(l)['dataset']=='hc2'];assert len(records)==128
 assert set(seal['files'])=={r['id']for r in records}
 for r in records:
  k=r['id'];s=seal['files'][k];p=MASKS/s.get('path',f'{k}.pt');assert sha(p)==s['sha256'];receipt=p.with_suffix('.json');rs=read(receipt);assert rs['GT_read'] is False and rs['sha256']==s['sha256'] and rs['native_selected_boxes_bit_exact'] is True
  if 'receipt_sha256'in s:assert sha(receipt)==s['receipt_sha256']
  oldsc=read(TRIAL/'student_cache'/f'{k}.json');assert sha(TRIAL/'student_cache'/f'{k}.pt')==oldsc['sha256']
  bound=rs.get('source_cache_sha256',rs.get('oldstudenthash',rs.get('old_student_sha256',rs.get('student_cache_sha256'))))
  assert bound==oldsc['sha256'],f'{k}: native mask input not bound to old student cache'
  assert sha(TRIAL/'predictions'/f'{k}.npz')==old['files'][k]['sha256']
 return seal,records

def unpack(pack):
 raw=pack.get('raw_out',pack.get('out'));assert raw is not None and all(k in raw for k in ['pred_masks','pred_boxes'])
 z=pack.get('z',pack.get('teacher',pack.get('native_result')));assert z is not None
 masks=raw['pred_masks'];boxes=raw['pred_boxes'];masks=masks[0]if masks.ndim==5 else masks;boxes=boxes[0]if boxes.ndim==4 else boxes
 assert masks.ndim==4 and masks.shape[0]==8 and boxes.ndim==3 and boxes.shape[0]==8
 idx=int(z.get('selected_query',pack.get('selected_query')));shape=z.get('input_shape',pack.get('input_shape'));assert shape and len(shape)==4 and shape[0]==8 and shape[1]==3
 # SingleRefer uses same-sized 8 images and size_divisibility=1. A saved input-mask audit is required.
 assert pack['nested_mask_all_false'] is True
 assert pack['nested_mask_shape']==[1,8,*shape[-2:]] and pack['nested_tensor_shape']==[1,*shape]
 assert torch.isfinite(masks).all() and torch.isfinite(boxes).all()
 # Official mask map convention: resize logits to resized image dimensions, then sigmoid.
 selected=F.interpolate(masks[:,idx,None].float(),size=tuple(shape[-2:]),mode='bilinear',align_corners=False).squeeze(1).sigmoid().numpy()
 assert np.isfinite(selected).all();return selected,arr(boxes),idx,z,list(masks.shape),shape

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run-after-seal',action='store_true');args=parser.parse_args();assert args.run_after_seal,'Preparation only; root must authorize --run-after-seal after 128 native mask seal'
 seal,records=validate_before_GT()
 labelsfile=ROOT/'data/evaluation/development_labels.jsonl';labelsha=sha(labelsfile);labels={x['id']:x for x in map(json.loads,labelsfile.read_text().splitlines())if x['dataset']=='hc2'}
 membership=read(OUT/'development_official_membership.json');members=set(membership['members']);changed=set(membership['different']);query=[];anchors=[];gt_count=0;all_shape_audits=[]
 for j,r in enumerate(records):
  k=r['id'];entry=seal['files'][k];pack=torch.load(MASKS/entry.get('path',f'{k}.pt'),map_location='cpu',weights_only=False);P,rawboxes,selected,z,native_shape,input_shape=unpack(pack)
  old=torch.load(TRIAL/'student_cache'/f'{k}.pt',map_location='cpu',weights_only=False);assert np.array_equal(arr(z['boxes']),arr(old['teacher']['boxes'])) and selected==old['teacher']['selected_query'];assert pack['source_cache_sha256']==sha(TRIAL/'student_cache'/f'{k}.pt') and pack['GT_read'] is False;positions=old['planned_positions'];ids=old['frame_ids'];assert pack['positions8']==positions and pack['frame_ids']==ids;meta=r['metadata'];gt=labels[k]['boxes'];scalevec=np.array([meta['width'],meta['height']]*2)
  q=dict(id=k,public_id=f'H{j+1:03}',in_official=k in members,official_boxes_changed=k in changed,anchors=[]);all_shape_audits.append(dict(native_shape=native_shape,input_shape=input_shape))
  for ai,position in enumerate(positions):
   frame=int(ids[position])
   if str(frame)not in gt:continue
   gt_count+=1;g=np.array(gt[str(frame)],dtype=float)/scalevec;gb=cxcywh(g);tb=arr(z['boxes'])[ai];eb=arr(old['E'])[position]
   soft=MaskScores(P[ai]);hard=MaskScores(P[ai]>.5);candidates=[]
   for base,b in [('Teacher',tb),('ERef',eb),('GT_posthoc',gb)]:
    for scale in SCALES:candidates.append(dict(name=f'{base}_s{scale:g}',base=base,scale=scale,box=np.r_[b[:2],b[2:]*scale],GT_generated=base=='GT_posthoc',duplicate=False))
   candidates.append(dict(name='fullframe',base='fullframe',scale=None,box=np.array([.5,.5,1,1]),GT_generated=False,duplicate=False))
   for qi,b in enumerate(rawboxes[ai]):
    if qi==selected:continue
    candidates.append(dict(name=f'nativequery_{qi}',base='native_other',scale=None,box=b,GT_generated=False,duplicate=biou(b,corners(tb))>=DUPLICATE_IOU,native_query=qi))
   for cand in candidates:
    b=cand.pop('box');v=soft.score(b);vh=hard.score(b);cand.update(box=b.tolist(),GT_IoU=biou(b,g),**v,pixel_Dice_hard=vh['pixel_Dice'],projection_Dice_hard=vh['projection_Dice'],geometry_reference=geometry_reward(b,tb))
   yy,xx=np.nonzero(P[ai]>.5);tight=np.array([xx.min()/soft.w,yy.min()/soft.h,(xx.max()+1)/soft.w,(yy.max()+1)/soft.h])if len(xx)else None
   components,n_components=connected_components(P[ai]>.5);counts=np.bincount(components.ravel());counts[0]=0;lcc=int(counts.argmax());ly,lx=np.nonzero(components==lcc)if lcc>0 else(np.array([]),np.array([]));lcc_tight=np.array([lx.min()/soft.w,ly.min()/soft.h,(lx.max()+1)/soft.w,(ly.max()+1)/soft.h])if len(lx)else None
   GTc=next(c for c in candidates if c['base']=='GT_posthoc'and c['scale']==1);Tc=next(c for c in candidates if c['base']=='Teacher'and c['scale']==1)
   predpool=[c for c in candidates if not c['GT_generated']];distinct=[c for c in predpool if not c['duplicate']];rho={}
   for label,pool in [('prediction_pool',predpool),('prediction_pool_distinct',distinct)]:
    rho[label]={}
    for score in RANK_KEYS:
     vx=np.array([c[score]for c in pool]);vy=np.array([c['GT_IoU']for c in pool]);rho[label][score]=float(spearmanr(vx,vy).statistic)if np.ptp(vx)>1e-12 and np.ptp(vy)>1e-12 else np.nan
   proj_inside=integrate1(soft.pxs,g[2]*soft.w)-integrate1(soft.pxs,g[0]*soft.w)+integrate1(soft.pys,g[3]*soft.h)-integrate1(soft.pys,g[1]*soft.h)
   row=dict(id=k,public_id=q['public_id'],anchor=ai,frame=frame,teacher_GT_IoU=Tc['GT_IoU'],ERef_GT_IoU=next(c for c in candidates if c['base']=='ERef'and c['scale']==1)['GT_IoU'],mask_empty_at_half=len(xx)==0,mask_probability_mean=float(P[ai].mean()),mask_probability_max=float(P[ai].max()),positive_pixel_fraction=float((P[ai]>.5).mean()),soft_mass_outside_GTbox_proxy=1-GTc['mask_coverage'],hard_mass_outside_GTbox_proxy=1-hard.score(gb)['mask_coverage'],soft_max_projection_mass_outside_GT_extent_proxy=1-proj_inside/max(soft.px.sum()+soft.py.sum(),1e-15),native_other_queries=len(rawboxes[ai])-1,native_other_duplicates=sum(c['base']=='native_other'and c['duplicate']for c in candidates),mask_tightbox=tight.tolist()if tight is not None else None,mask_tightbox_GT_IoU=biou(cxcywh(tight),g)if tight is not None else 0.,mask_tightbox_teacher_IoU=biou(cxcywh(tight),corners(tb))if tight is not None else 0.,mask_tight_extent_over_GT=((tight[2:]-tight[:2])/(g[2:]-g[:2])).tolist()if tight is not None else None,components_at_half=int(n_components),largest_component_fraction=float(len(lx)/len(xx))if len(xx)else 0.,largest_component_tight_GT_IoU=biou(cxcywh(lcc_tight),g)if lcc_tight is not None else 0.,rankings=rho,candidates=candidates)
   if row['mask_empty_at_half']:row['hard_mass_outside_GTbox_proxy']=np.nan
   anchors.append(row);q['anchors'].append(row)
  query.append(q);del P,old,pack;gc.collect()
  if (j+1)%16==0:print('mask_diagnostic_CPU',j+1,flush=True)
 assert gt_count==425 and sha(labelsfile)==labelsha and not torch.cuda.is_initialized()
 def qmacro(func):return summary([np.nanmean([func(a)for a in q['anchors']])for q in query if q['anchors']])
 agg=dict(queries=128,GT_anchor_count=gt_count,queries_with_GT_anchors=sum(bool(q['anchors'])for q in query),no_segmentation_GT='GT is rectangle only; outside-GTbox mass is a proxy, not segmentation precision or identity proof; hard-empty masks have undefined hard outside mass and are reported separately',mean_teacher_box_GT_IoU=qmacro(lambda a:a['teacher_GT_IoU']),mask_tightbox_GT_IoU=qmacro(lambda a:a['mask_tightbox_GT_IoU']),mask_tightbox_teacher_IoU=qmacro(lambda a:a['mask_tightbox_teacher_IoU']),empty_mask_fraction=qmacro(lambda a:float(a['mask_empty_at_half'])),soft_mass_outside_GTbox_proxy=qmacro(lambda a:a['soft_mass_outside_GTbox_proxy']),hard_mass_outside_GTbox_proxy=qmacro(lambda a:a['hard_mass_outside_GTbox_proxy']),soft_max_projection_mass_outside_GT_extent_proxy=qmacro(lambda a:a['soft_max_projection_mass_outside_GT_extent_proxy']),mask_components_at_half=qmacro(lambda a:a['components_at_half']),largest_component_fraction=qmacro(lambda a:a['largest_component_fraction']),largest_component_tight_GT_IoU=qmacro(lambda a:a['largest_component_tight_GT_IoU']),native_other_duplicate_fraction=qmacro(lambda a:a['native_other_duplicates']/a['native_other_queries']),rankings={label:{score:qmacro(lambda a,l=label,s=score:a['rankings'][l][s])for score in RANK_KEYS}for label in ['prediction_pool','prediction_pool_distinct']},bias_by_base={},version_groups={})
 for base in ['Teacher','ERef','GT_posthoc']:
  out={}
  for score in SCORE_KEYS:
   def facts(a):
    cc=[c for c in a['candidates']if c['base']==base];orig=next(c for c in cc if c['scale']==1);small=max(c[score]for c in cc if c['scale']<1);large=max(c[score]for c in cc if c['scale']>1);full=next(c for c in a['candidates']if c['base']=='fullframe');return dict(smaller_beats_base=float(small>orig[score]+1e-10),larger_beats_base=float(large>orig[score]+1e-10),fullframe_beats_base=float(full[score]>orig[score]+1e-10),max_gain_smaller=small-orig[score],fullframe_gain=full[score]-orig[score])
   out[score]={f:qmacro(lambda a,f=f:facts(a)[f])for f in ['smaller_beats_base','larger_beats_base','fullframe_beats_base','max_gain_smaller','fullframe_gain']}
  agg['bias_by_base'][base]=out
 for group,pred in [('official_same',lambda q:q['in_official']and not q['official_boxes_changed']),('official_changed',lambda q:q['official_boxes_changed']),('not_official',lambda q:not q['in_official'])]:
  qq=[q for q in query if pred(q)];agg['version_groups'][group]=dict(queries=len(qq),GTanchors=sum(len(q['anchors'])for q in qq),teacher_IoU=summary([np.mean([a['teacher_GT_IoU']for a in q['anchors']])for q in qq if q['anchors']]),mask_tightbox_GT_IoU=summary([np.mean([a['mask_tightbox_GT_IoU']for a in q['anchors']])for q in qq if q['anchors']]))
 result=dict(status='posthoc_native_mask_CPU_diagnostic_not_method_selection',prespecified=dict(scales=SCALES,duplicates_threshold=DUPLICATE_IOU,scores='soft mean / soft mass coverage / softmask-rectangle Dice / max-axis-projection softDice; hard threshold=.5 fixed reference',GT_generated_candidates='GT scales separated from prediction-only ranking',mapping='F.interpolate logits to original resized input_shape H,W, bilinear align_corners=False; then sigmoid, no input padding required',integration='continuous fractional-pixel box area and exact piecewise-constant softmask integral',no_temperature_no_adaptation=True),proof=dict(native_seal_sha256=sha(MASKS/'sealed.json'),development_labels_sha256=labelsha,all128_hashes_before_GT=True,no_CUDA_initialized=True),aggregate=agg,shape_audits=all_shape_audits,queries=query)
 (OUT/'mask_completeness.json').write_text(json.dumps(result,indent=2,allow_nan=True)+'\n');(OUT/'mask_completeness_public_summary.json').write_text(json.dumps(dict(status=result['status'],prespecified=result['prespecified'],aggregate=agg),indent=2,allow_nan=True)+'\n');print(json.dumps(agg,indent=2))

if __name__=='__main__':main()
