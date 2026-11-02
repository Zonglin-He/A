"""Independent old336 post-seal CPU audit; never reads GT boxes or FINAL."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
import sys,json,hashlib,re,ast,time,datetime,argparse
from pathlib import Path
import numpy as np
from scipy.special import softmax
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];EXP=ROOT/'research/round48/experiment';CAP=ROOT/'crossplatform/round24/capture336_attempt001'
from official_span_independent48 import dictionary_from_source,span_pair_masks
ARMS=['current_proxy_CPU64','official_LITERAL_indices_DIAGNOSTIC','text_only_subject_parser','official_charspan_REMAP_DIAGNOSTIC','cached_native_current_BRIDGE']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(d):
 s=read(d/'SEALED.json')
 for n,h in s['files'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts and sha(d/n)==h,n
 return s
def verify_independent_code():
 lock=read(HERE/'INDEPENDENT_CODE_LOCK.json')
 for rel,h in lock['program_sha256'].items():assert sha(ROOT/rel)==h,rel
def cohort():
 rr=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');membership={}
 for split in ['D208','V64','M64']:
  for r in read(ROOT/'research/round24/input'/(split+'_PRIVATE.json')):assert r['id'] not in membership;membership[r['id']]=split
 assert len(rr)==len(membership)==336 and set(membership)=={r['id'] for r in rr};return rr,membership
def old_index(rr):
 import ijson
 prep=read(ROOT/'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json');p=ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json';assert sha(p)==prep['scoring_index_sha256'];wanted={r['id']:r for r in rr};out={}
 with p.open('rb') as f:
  for x in ijson.items(f,'item'):
   if x['id'] not in wanted:continue
   r=wanted[x['id']];assert x['platform']==r['platform'] and x['scene']==r['scene'] and Path(x['meta_path']).resolve()==Path(r['metapath']).resolve() and type(x['object_index']) is int and x['object_index']>=0
   out[x['id']]={k:x[k] for k in ['id','platform','scene','meta_path','object_index','metadata_container_sha256']}
 assert set(out)==set(wanted);return out

def caption_classes(rr,index):
 import ijson
 grouped={};out={}
 for r in rr:grouped.setdefault(r['metapath'],[]).append(r)
 for p,rs in grouped.items():
  expected={index[r['id']]['metadata_container_sha256'] for r in rs};assert len(expected)==1 and sha(p)==next(iter(expected));targets={index[r['id']]['object_index'] for r in rs};fields={};j=-1
  with Path(p).open('rb') as f:
   for prefix,event,value in ijson.parse(f):
    if prefix=='ground_info.item' and event=='start_map':j+=1
    elif j in targets and prefix in ['ground_info.item.class','ground_info.item.caption'] and event=='string':fields.setdefault(j,{})[prefix.rsplit('.',1)[1]]=value
  for r in rs:
   x=fields[index[r['id']]['object_index']];assert set(x)=={'caption','class'} and ' '.join(x['caption'].replace(',',' ,').split())==r['caption'];out[r['id']]=x['class'].lower()
 return out

def independent_head(caption,dep):
 """Independent reading of frozen dependency-role priority, not NLP re-inference."""
 source=(ROOT/'crossplatform/round48/parser48.py').read_text();meta=set(dictionary_from_source(source,'META'));directives=set(dictionary_from_source(source,'DIRECTIVES'))
 nominal=lambda w:w.get('upos') in {'NOUN','PROPN'} and str(w.get('lemma') or w.get('text','')).casefold() not in meta
 def located(w):
  a,b=w.get('start_char'),w.get('end_char');return type(a) is int and type(b) is int and 0<=a<b<=len(caption) and caption[a:b]==w.get('text')
 def selected(words,w):
  if not located(w) or any(v.get('head')==w['id'] and str(v.get('deprel','')).split(':')[0]=='conj' and nominal(v) for v in words):return None
  return [w['start_char'],w['end_char']]
 ss=dep.get('sentences',[]);markers=list(re.finditer(r'\bobject\s+type\s*:',caption,re.I))
 if len(markers)>1:return None
 if any(w.get('upos') in {'NOUN','PROPN'} and not located(w) for sent in ss for w in sent['words']):return None
 if markers:
  left=markers[0].end()
  stops=[len(caption)]+[left+m.start() for m in re.finditer(r'[;\n\r]',caption[left:])]+[left+m.start() for m in re.finditer(r'\b(?:main\s+appearance\s+features|position\s+information|relationship\s+with(?:\s+the)?\s+surrounding\s+environment)\s*:',caption[left:],re.I)]
  right=min(stops);pool=[]
  for sent in ss:
   words=sent['words'];inside={w['id']:w for w in words if located(w) and left<=w['start_char']<w['end_char']<=right}
   for w in inside.values():
    if not nominal(w):continue
    j=w['head'];seen=set();has_nominal=False
    while j in inside and j not in seen:
     seen.add(j);a=inside[j]
     if a.get('upos') in {'NOUN','PROPN'}:has_nominal=True;break
     j=a['head']
    if not has_nominal:pool.append((words,w))
  return selected(*pool[0]) if len(pool)==1 else None
 for sent in ss:
  words=sent['words'];roots=[w for w in words if w.get('head')==0]
  if len(roots)!=1:
   if any(nominal(w) for w in words):return None
   continue
  root=roots[0];subjects=[w for w in words if w.get('head')==root['id'] and str(w.get('deprel','')).split(':')[0] in {'nsubj','csubj'}];heads=[w for w in subjects if nominal(w)]
  if len(heads)>1:return None
  if heads:return selected(words,heads[0])
  if any(str(w['deprel']).split(':')[0]=='csubj' for w in subjects):return None
  if nominal(root):return selected(words,root)
  imperative=root.get('upos')=='VERB' and str(root.get('lemma') or root.get('text','')).casefold() in directives and all(w.get('upos')=='PRON' and str(w.get('lemma') or w.get('text','')).casefold()=='you' for w in subjects)
  if imperative:
   objs=[w for w in words if w.get('head')==root['id'] and str(w.get('deprel','')).split(':')[0]=='obj' and nominal(w)];return selected(words,objs[0]) if len(objs)==1 else None
  if any(nominal(w) for w in words):return None
 return None

def head_contract(caption,dep,case):
 """Structural caption-only verification, distinct from semantic identity audit."""
 if case['status']=='UNKNOWN':assert case['head_span'] is None and case['head_text'] is None;return {'missing_word_offsets':sum(type(w.get('start_char')) is not int or type(w.get('end_char')) is not int for s in dep['sentences'] for w in s['words'])}
 assert case['status']=='FOUND';si=case['sentence_index'];words=dep['sentences'][si]['words'];head=next(w for w in words if w['id']==case['head_word_id']);lo,hi=case['head_span'];assert head['upos'] in ['NOUN','PROPN'] and [head['start_char'],head['end_char']]==[lo,hi] and caption[lo:hi]==head['text']==case['head_text'];assert not any(w.get('head')==head['id'] and str(w.get('deprel','')).split(':')[0]=='conj' and w.get('upos') in ['NOUN','PROPN'] and str(w.get('lemma') or w.get('text','')).casefold() not in set(dictionary_from_source((ROOT/'crossplatform/round48/parser48.py').read_text(),'META')) for w in words)
 kind=case['selection_kind'];byid={w['id']:w for w in words};roots=[w for w in words if w['head']==0]
 if kind=='explicit_object_type_head':
  a,b=case['field_span'];assert 0<=a<=lo<hi<=b<=len(caption);parent=head['head'];seen=set()
  while parent in byid and parent not in seen:
   seen.add(parent);ancestor=byid[parent];al,ah=ancestor.get('start_char'),ancestor.get('end_char')
   if type(al) is int and type(ah) is int and a<=al<ah<=b:assert ancestor['upos'] not in ['NOUN','PROPN'];parent=ancestor['head']
   else:break
 elif kind in ['matrix_nsubj','matrix_csubj']:assert len(roots)==1 and head['head']==roots[0]['id'] and str(head['deprel']).split(':')[0]==kind.removeprefix('matrix_')
 elif kind=='root_nominal_or_deictic_copular':assert head['head']==0
 elif kind=='imperative_direct_object':assert len(roots)==1 and head['head']==roots[0]['id'] and head['deprel'].split(':')[0]=='obj' and str(roots[0]['lemma']).casefold() in {'find','locate','identify','select','choose','show','pick','spot'}
 else:raise AssertionError(kind)
 return {'missing_word_offsets':sum(type(w.get('start_char')) is not int or type(w.get('end_char')) is not int for s in dep['sentences'] for w in s['words'])}
def main():
 start=time.perf_counter();verify_independent_code();auth=read(EXP/'ROOT_INDEPENDENT_AUDIT_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_R48_INDEPENDENT_CPU_AUDIT_NO_GTBOX_NO_FINAL';plan=read(EXP/'PLAN_LOCK.json');assert auth['plan_sha256']==sha(EXP/'PLAN_LOCK.json');P=ROOT/plan['prediction_output'];D=ROOT/plan['parse_output'];assert auth['prediction_seal_sha256']==sha(P/'SEALED.json');ps=verify(P);ds=verify(D);assert ps['parser_seal_sha256']==sha(D/'SEALED.json') and ps['created_utc']>=ds['created_utc'];assert ps['n']==ds['n']==336 and ps['arms']==ARMS
 for n,h in {**plan['program_sha256'],**plan['input_sha256'],**plan['manifest_sha256']}.items():assert sha(ROOT/n)==h,n
 rr,membership=cohort();index=old_index(rr);classes=caption_classes(rr,index);source=read(CAP/'SEALED.json');assert sha(CAP/'SEALED.json')==plan['source_cache_seal_sha256'];synsource=(ROOT/'crossplatform/round3/p0/upstream/src/joint_det_dataset.py').read_text();syn=dictionary_from_source(synsource,'M3ED_SYNONYMS');lex=dictionary_from_source((ROOT/'crossplatform/round7/regression_adapter.py').read_text(),'LEXICON')
 from transformers import RobertaTokenizerFast
 tok=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True);maxp=maxscore=0.;auditrows=[];bridgediffs=[]
 for r in rr:
  rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(CAP/rel)==source['files'][rel];z=load(CAP/rel);p=load(P/'predictions_PRIVATE'/(r['id']+'.npz'));case=read(P/'cases_PRIVATE'/(r['id']+'.json'));parsed=read(D/'cases_PRIVATE'/(r['id']+'.json'));dep=read(D/'dependencies_PRIVATE'/(r['id']+'.json'));assert case['parser']==parsed and parsed['caption_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest() and parsed['parser_uses_metadata_class_or_boxes'] is False
  assert independent_head(r['caption'],dep)==parsed['head_span'],r['alias'];hc=head_contract(r['caption'],dep,parsed);encoded=tok(r['caption']+' . not mentioned',return_offsets_mapping=True);offset=np.asarray(encoded['offset_mapping']);T=len(z['tokens']);assert len(offset)==T and np.array_equal(offset,p['actual_offsets']);m=re.search(r'\b(?:'+'|'.join(lex)+r')\b',r['caption']+' . not mentioned',re.I);span=(m.start(),m.end()) if m else None;pm=((offset[:,1]>span[0])&(offset[:,0]<span[1])&(offset[:,1]>offset[:,0])) if span else offset[:,1]>offset[:,0];assert np.array_equal(pm,z['token_mask']);tm=np.zeros(T,bool)
  if parsed['status']=='FOUND':a,b=parsed['head_span'];tm=(offset[:,1]>a)&(offset[:,0]<b)&(offset[:,1]>offset[:,0])
  tf=not tm.any();tm=pm.copy() if tf else tm
  literal,remap,detail=span_pair_masks(r['caption'],classes[r['id']],syn,tok,T);lm=np.zeros(T,bool) if literal is None else literal[:T];rm=np.zeros(T,bool) if remap is None else remap[:T];lf=not lm.any();rf=not rm.any();lm=pm.copy() if lf else lm;rm=pm.copy() if rf else rm;mask=np.stack([pm,lm,tm,rm]);assert np.array_equal(mask,p['positive_masks']);assert (tf,lf,rf)==(case['parser_fallback_to_current'],case['official_literal_fallback_independent'],case['official_remap_fallback_independent'])
  if detail['status']=='matched':assert list(detail['official_charspan'])==case['official']['span_formatted'] and list(detail['actual_charspan'])==case['official']['span_actual']
  probabilities=softmax(z['query'].astype(float)@z['tokens'].astype(float).T/.07,axis=-1);pe=float(abs(probabilities-p['shared_CPU64_probabilities']).max());assert pe<2e-15;maxp=max(maxp,pe);expected=np.stack([probabilities[:,v].sum(-1) for v in mask]+[z['scores'].astype(float)]);se=float(abs(expected-p['scores']).max());assert se<1e-14;maxscore=max(maxscore,se);assert p['arm_names'].tolist()==ARMS and np.array_equal(p['queries'],p['scores'].argmax(-1)) and np.array_equal(expected.argmax(-1),p['queries']);assert np.array_equal(p['boxes'],z['boxes'][p['queries']]) and np.array_equal(p['source_boxes'],z['boxes']) and np.array_equal(p['stored_native_current'],z['scores']);assert np.array_equal(p['scores'][4],z['scores'].astype(float))
  row={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'split':membership[r['id']],'parser_status':parsed['status'],'structured':parsed['structured'],'parser_reason':parsed['reason'],'parser_fallback':tf,'official_no_match':detail['status']!='matched','official_literal_fallback':lf,'official_remap_fallback':rf,'literal_remap_mask_equal':bool(np.array_equal(mask[1],mask[3])),'current_parser_mask_equal':bool(np.array_equal(pm,tm)),'queries':dict(zip(ARMS,map(int,p['queries']))),'top_tie_counts':{a:int((p['scores'][j]==p['scores'][j].max()).sum()) for j,a in enumerate(ARMS)},'CPU64_vs_storednative_maxerror':float(abs(p['scores'][0]-z['scores']).max()),**hc};auditrows.append(row)
  if row['queries'][ARMS[0]]!=row['queries'][ARMS[4]]:bridgediffs.append(row)
 dump(HERE/'ALL336_READOUT_AUDIT_ROWS_PRIVATE.json',auditrows);dump(HERE/'NATIVE_BRIDGE_DIFFERENCES_PRIVATE.json',bridgediffs);result={'state':'PASS_R48_INDEPENDENT_CPU_ARRAY_SPAN_AUDIT_NO_GTBOX','n':336,'arms':5,'selected_boxes_exact':1680,'probability_maxerror':maxp,'score_maxerror':maxscore,'sourcequery_tokens_boxes_unchanged':True,'historical_input_ids_not_stored':True,'historical_input_id_identity_claimed':False,'parser_semantic_correctness_not_certified_by_structural_audit':True,'parser_UNKNOWN':sum(x['parser_status']=='UNKNOWN' for x in auditrows),'missing_dependency_word_offsets':sum(x['missing_word_offsets'] for x in auditrows),'native_bridge_query_differences':len(bridgediffs),'official_literal_remap_mask_differences':sum(not x['literal_remap_mask_equal'] for x in auditrows),'GTclass_only_two_official_diagnostics':True,'GTbbox_or_FINAL_read':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(HERE/'AUDIT_RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
