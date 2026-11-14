"""Caption-owned soft observations. No labels, experts or model selection."""
import re,hashlib,itertools
import numpy as np
COLORS=['red','orange','yellow','green','cyan','blue','purple','pink','brown','beige','white','gray','black']
STOP=set('side front back rear left right view image picture observer viewer center centre middle edge corner position information appearance feature features environment type line lines body roof rack taillight taillights shirt top shorts pants jacket sleeve hand arm leg head face wheel window color colour light'.split())
def parse85(caption,document,subject):
 if subject['status']!='FOUND':return {'usable':False,'entities':[],'relations':[],'reason':'subject_unknown'}
 words=[dict(w,sentence=si) for si,s in enumerate(document['sentences']) for w in s['words'] if isinstance(w.get('start_char'),int) and isinstance(w.get('end_char'),int)]
 headspan=subject['head_span'];target=next((w for w in words if [w['start_char'],w['end_char']]==headspan),None)
 if target is None:return {'usable':False,'entities':[],'relations':[],'reason':'subject_offset_unknown'}
 def phrase(w):
  members=[w];todo=[w['id']]
  while todo:
   hd=todo.pop();cs=[x for x in words if x['sentence']==w['sentence'] and x['head']==hd and x['deprel'].split(':')[0] in {'amod','compound','det','nummod','flat','fixed'} and x not in members];members+=cs;todo +=[x['id'] for x in cs]
  lo=min(x['start_char'] for x in members);hi=max(x['end_char'] for x in members);return lo,hi,caption[lo:hi]
 lo,hi,txt=phrase(target);attrtext=txt
 if subject.get('structured'):
  m=re.search(r'\bmain\s+appearance\s+features\s*:\s*([^;\n]+)',caption,re.I)
  if m:attrtext=m.group(1);txt+=' '+attrtext
 ent=[{'role':'T','phrase':txt,'span':[lo,hi],'attributes':color_words(attrtext),'head':target['text']}]
 for w in words:
  if w['upos'] not in {'NOUN','PROPN'} or w['text'].lower() in STOP or w['deprel'].split(':')[0] in {'compound','amod'} or [w['start_char'],w['end_char']]==headspan:continue
  a,b,t=phrase(w)
  if a<hi or (subject.get('structured') and a<caption.lower().find('relationship with')):continue
  if any(e['span']==[a,b] for e in ent):continue
  ent.append({'role':'R'+str(len(ent)-1),'phrase':t,'span':[a,b],'attributes':color_words(t),'head':w['text']})
  if len(ent)==5:break
 edges=[];camera=bool(re.search(r'\b(view|image|picture|observer|viewer|screen|frame)\b',caption,re.I))
 patt=re.compile(r'\b(left\s+of|right\s+of|in\s+front\s+of|behind|next\s+to|beside|near)\b',re.I)
 for a,b in itertools.combinations(ent,2):
  between=caption[a['span'][1]:b['span'][0]];matches=list(patt.finditer(between))
  if matches:
   match=matches[-1];predicate={'left of':'LEFT','right of':'RIGHT','in front of':'FRONT','behind':'BEHIND','next to':'NEAR','beside':'NEAR','near':'NEAR'}[re.sub(r'\s+',' ',match.group(1).lower())]
   # An entity pair is directed by literal noun ordering, never by GT.
   known=camera and not re.search(r"\b(its|their|own)\b",between,re.I)
   edges.append({'subject':a['role'],'reference':b['role'],'predicate':predicate,'known_frame':known})
 for n,e in enumerate(ent[1:]):
  end=ent[n+2]['span'][0] if n+2<len(ent) else len(caption);tail=caption[e['span'][1]:end]
  m=re.search(r'\b(?:on|to|at)\s+(?:the\s+)?(left|right)(?!\s+of)\b',tail,re.I)
  if m:edges.append({'subject':e['role'],'reference':'T','predicate':m.group(1).upper(),'known_frame':camera and not bool(re.search(r'\bits\b',tail,re.I))})
 return {'usable':True,'entities':ent,'relations':edges,'reason':'caption_dependency_and_literal_direction'}
def color_words(s):
 if re.search(r'\b(not|no|without|except)\b',s,re.I):return []
 return [c for c in COLORS if re.search(r'\b'+c+r'\b',s,re.I)]
def groups85(boxes,xyz):
 support=(np.abs(xyz[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(-1);labels=np.full(256,-1,int);reps=[]
 for i in range(256):
  for k,j in enumerate(reps):
   shared=(support[i]&support[j]).sum();union=(support[i]|support[j]).sum();lo=np.maximum(boxes[i,:3]-boxes[i,3:]/2,boxes[j,:3]-boxes[j,3:]/2);hi=np.minimum(boxes[i,:3]+boxes[i,3:]/2,boxes[j,:3]+boxes[j,3:]/2);inter=np.maximum(0,hi-lo).prod();iou=inter/max(1e-12,boxes[i,3:].prod()+boxes[j,3:].prod()-inter)
   if iou>=.9 or (iou>=.5 and shared>=2 and shared/max(1,union)>=.6):labels[i]=k;break
  if labels[i]<0:labels[i]=len(reps);reps.append(i)
 return labels,np.array(reps),support
def measurements85(boxes,xyz,g,desc,rect,rectvalid):
 uv=g['uv_grid']/[48,32];vis=g['visible'];cell=g['cell'];K=len(COLORS);means=np.zeros((3,256,K));n=np.zeros((3,256));coverage=np.zeros((3,256));interval=np.zeros((256,2));reliability=np.zeros(256)
 for q,b in enumerate(boxes):
  for t,scale in enumerate([.85,1.,1.15]):
   ix=np.flatnonzero(vis&(np.abs(xyz-b[:3])<=b[3:]*scale/2).all(-1));cells=np.unique(cell[ix]);n[t,q]=len(cells)
   if len(cells):means[t,q]=desc[cells].mean(0)
   if len(ix)>1 and rectvalid[q]:coverage[t,q]=np.minimum(1,np.ptp(uv[ix],axis=0)/np.maximum(rect[q,2:]-rect[q,:2],1e-3)).min()
   if t==1 and len(ix) and rectvalid[q]:
    a,bx=np.quantile(uv[ix,0],[.1,.9]);center=(rect[q,0]+rect[q,2])/2;interval[q]=[min(a,center),max(bx,center)];reliability[q]=len(cells)/(len(cells)+8)*min(1.,coverage[t,q]/.3)
 return means,n,coverage,interval,reliability
def attribute85(means,n,cover,request):
 if len(request)!=1:return np.zeros(256)
 c=request[0];k=COLORS.index(c);hue={'red':0.,'orange':.08,'yellow':.16,'green':.33,'cyan':.5,'blue':.66,'purple':.78,'pink':.91};alt=[]
 for j,v in enumerate(COLORS):
  if c in hue and v in hue and min(abs(hue[v]-hue[c]),1-abs(hue[v]-hue[c]))>=.24:alt.append(j)
  if (c,v) in [('white','black'),('black','white')]:alt.append(j)
 if not alt:return np.zeros(256)
 strength=means[:,:,alt].max(-1).min(0);wanted=means[:,:,k].max(0);reliability=n.min(0)/(n.min(0)+8)*np.minimum(1.,cover.min(0)/.3);stable=np.clip(1-means[:,:,alt].max(-1).ptp(0),0,1) if False else np.clip(1-np.ptp(means[:,:,alt].max(-1),axis=0),0,1)
 return reliability*stable*np.maximum(0,strength-wanted)
def evidence85(graph,entity_scores,measure,labels,reps,variant='real',seedkey=''):
 means,n,cover,interval,rel=measure;cost=np.zeros(256);details=[]
 if not graph['usable']:return cost,details
 attrs=np.stack([attribute85(means,n,cover,e['attributes']) for e in graph['entities']]);cost+=attrs[0]
 for e in graph['relations']:
  if not e['known_frame'] or 'T' not in [e['subject'],e['reference']]:continue
  target_first=e['subject']=='T';rrole=e['reference'] if target_first else e['subject'];roles=[x['role'] for x in graph['entities']]
  if rrole not in roles:continue
  ri=roles.index(rrole);rs=entity_scores[ri]-attrs[ri];group_logit=np.array([np.logaddexp.reduce(rs[labels==j])-np.log((labels==j).sum()) for j in range(len(reps))]);rprob=np.exp(group_logit-group_logit.max());rprob/=rprob.sum()
  lo,hi=interval.T;tl=lo[:,None];th=hi[:,None];rl=lo[reps][None];rh=hi[reps][None];pred=e['predicate']
  if not target_first:tl,rl=rl,tl;th,rh=rh,th
  if pred=='LEFT':violation=np.maximum(0,tl-rh-.01)
  elif pred=='RIGHT':violation=np.maximum(0,rl-th-.01)
  else:continue  # NEAR/depth/intrinsic frames unknown in this bounded version.
  reliability=np.sqrt(rel[:,None]*rel[reps][None]);same=labels[:,None]==np.arange(len(reps))[None];penalty=reliability*np.tanh(violation/.1);penalty[same]=0
  # Same-hypothesis reference mass is neutral UNKNOWN; never renormalized away.
  good=np.exp(-penalty/.25);marginal=good@rprob;term=-.25*np.log(np.maximum(marginal,1e-12));cost+=term;details.append({'role':rrole,'predicate':pred,'nonzero_pair_costs':int((penalty>0).sum()),'prior_mass_same_group_mean':float(np.mean(rprob[labels]))})
 if variant=='mismatch':cost=cost[np.random.default_rng(int(hashlib.sha256(('R85costshuffle|'+seedkey).encode()).hexdigest()[:16],16)).permutation(256)]
 return cost,details
