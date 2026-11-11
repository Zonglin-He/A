from common72 import *
from text72 import COLORS
from scipy.special import softmax,logsumexp
HUES={'red':0.,'orange':.08,'yellow':.16,'green':.33,'cyan':.5,'blue':.66,'purple':.78,'pink':.91}
def negative(measure,req,box=False,shuffle=False,sid=''):
 known=measure['box_known' if box else 'known'].copy();low=measure['box_descriptor' if box else 'descriptor'].copy();high=measure['box_descriptor' if box else 'high_descriptor'].copy()
 if shuffle:
  ix=np.flatnonzero(known);pp=perm(len(ix),'R72_RGB_component_binding|'+sid);low[ix]=low[ix][pp];high[ix]=high[ix][pp]
 color=req['color'];out=np.zeros(len(known),bool)
 if color not in COLORS:return out
 # Whole-object absence and achromatic windows/shadows never certify a contradiction.
 if color in ['black','white','gray'] and req['scope']=='whole':return out
 alts=[]
 for c in COLORS:
  if color in HUES and c in HUES and min(abs(HUES[c]-HUES[color]),1-abs(HUES[c]-HUES[color]))>=.24:alts.append(COLORS.index(c))
  if req['scope']!='whole' and (color,c) in [('black','white'),('white','black')]:alts.append(COLORS.index(c))
 if not alts:return out
 wanted=high[:,COLORS.index(color)];opposite=low[:,alts].max(1);return known&(opposite>=.55)&(wanted<=.15)&(opposite-wanted>=.35)

def relation_mask(a,boxes,tm,rm,parsed,qref):
 ng=len(a['rects']);out=np.zeros(ng,bool);record={'reference_posterior_available':qref is not None,'anchor_mass':0.,'anchor_groups':0,'relation_witness_used':False,'reference_identity_certified':False}
 if parsed['relation'] is None or qref is None or parsed['reference']['color'] not in COLORS:return out,record
 pg=np.bincount(a['query_group'],weights=qref,minlength=ng);order=np.argsort(-pg,kind='stable');k=int(np.searchsorted(np.cumsum(pg[order]),.95))+1;anchors=order[:k];record.update(anchor_mass=float(pg[anchors].sum()),anchor_groups=len(anchors));rcol=COLORS.index(parsed['reference']['color']);anchor_ok=rm['known'][anchors]&(rm['descriptor'][anchors,rcol]>=.35)
 if not anchor_ok.all() or not len(anchors):return out,record
 # Full projected box intervals, not visible-surface centers, establish opposite order.
 rect=a['rects'];w=max(float(a.get('image_width',np.array(1))),1);good=tm['known']&(a['project_reliability']>=.99)
 for g in np.flatnonzero(good):
  if g in anchors:continue
  samephysical=[]
  for j in anchors:
   b1=boxes[a['group_representatives'][g]];b2=boxes[a['group_representatives'][j]];lo=np.maximum(b1[:3]-b1[3:]/2,b2[:3]-b2[3:]/2);hi=np.minimum(b1[:3]+b1[3:]/2,b2[:3]+b2[3:]/2);inter=np.maximum(hi-lo,0).prod();samephysical.append(inter/max(min(b1[3:].prod(),b2[3:].prod()),1e-9)>.1)
  if any(samephysical):continue
  if parsed['relation']=='left':opposite=rect[g,0]>rect[anchors,2]+.02*w
  else:opposite=rect[g,2]<rect[anchors,0]-.02*w
  out[g]=bool(opposite.all())
 record['relation_witness_used']=bool(out.any());return out,record

def build(a,boxes,target,ref,parsed,qref,sid):
 color=negative(target,parsed['target']);rgb=negative(target,parsed['target'],shuffle=True,sid=sid);box=negative(target,parsed['target'],box=True);rel,rec=relation_mask(a,boxes,target,ref,parsed,qref);N=color|rel;obs=target['known'];shuf=N.copy();ix=np.flatnonzero(obs);shuf[ix]=N[ix][perm(len(ix),'R72_negative_budget|'+sid)]
 assert shuf.sum()==N.sum() and not shuf[~obs].any()
 return {'main_N':N,'RGB_N':rgb|rel,'Nshuffle_N':shuf,'box_N':box|rel,'color_N':color,'relation_N':rel},rec

def direct(p0,N,kind='opt'):
 if not N.any() or N.all():return p0.copy()
 a=float(p0[N].sum())
 if kind=='mu1':return softmax(np.log(np.maximum(p0,1e-300))-N.astype(float))
 b=0. if kind=='hard' else a/2
 q=p0.copy();q[N]*=b/max(a,1e-300);q[~N]*=(1-b)/max(1-a,1e-300);return q/q.sum()
