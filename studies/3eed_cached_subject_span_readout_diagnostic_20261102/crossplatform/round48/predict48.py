"""Five fixed source-cache readouts. Class diagnostics only after parser seal."""
import sys,time,re,ast
import numpy as np
from common48 import *
ARMS=['current_proxy_CPU64','official_LITERAL_indices_DIAGNOSTIC','text_only_subject_parser','official_charspan_REMAP_DIAGNOSTIC','cached_native_current_BRIDGE']
def proxy_span(caption):
 tree=ast.parse((ROOT/'crossplatform/round7/regression_adapter.py').read_text());lexicon=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LEXICON' for t in n.targets));m=re.search(r'\b(?:'+'|'.join(lexicon)+r')\b',caption+' . not mentioned',re.I);return [m.start(),m.end()] if m else None
def span_mask(offsets,span):
 if span is None:return np.zeros(len(offsets),bool)
 return (offsets[:,1]>span[0])&(offsets[:,0]<span[1])&(offsets[:,1]>offsets[:,0])
def main():
 a=args();plan,rows,membership,auth=context(a);parsed=ROOT/plan['parse_output'];ps=verify(parsed);assert ps['state']=='R48_CAPTION_ONLY_PARSER_ALL336_SEALED_BEFORE_METADATA_CLASS_READ' and ps['n']==336
 assert auth['allow_official_class_only_after_parser_seal'] is True
 # No metadata helper import/call before the complete caption-only seal above.
 from official48 import read_class_allowlist,official_masks
 classes,class_receipts=read_class_allowlist(rows)
 from transformers import RobertaTokenizerFast
 tok=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True)
 out=ROOT/plan['prediction_output'];out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();source=ROOT/'crossplatform/round24/capture336_attempt001';ss=read(source/'SEALED.json');assert sha(source/'SEALED.json')==plan['source_cache_seal_sha256'];start=time.perf_counter();bridge=[]
 for i,r in enumerate(rows):
  rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(source/rel)==ss['files'][rel];z=npz(source/rel);T=len(z['tokens']);actual=r['caption']+' . not mentioned';encoded=tok(actual,return_offsets_mapping=True);offset=np.asarray(encoded['offset_mapping'],int);assert len(offset)==T<=256
  proxy=proxy_span(r['caption']);pm=span_mask(offset,proxy) if proxy else offset[:,1]>offset[:,0];assert pm.any() and np.array_equal(pm,z['token_mask'])
  raw=read(parsed/'cases_PRIVATE'/(r['id']+'.json'));assert raw['caption_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest();tm=span_mask(offset,raw['head_span']) if raw['status']=='FOUND' else np.zeros(T,bool);parser_fallback=not tm.any();tm=pm.copy() if parser_fallback else tm
  literal,remap,official=official_masks(r['caption'],classes[r['id']],tok,T,r['platform']);lf=not literal.any();rf=not remap.any();literal=pm.copy() if lf else literal;remap=pm.copy() if rf else remap
  logits=np.asarray(z['query'],np.float64)@np.asarray(z['tokens'],np.float64).T/.07;logits-=logits.max(-1,keepdims=True);p=np.exp(logits);p/=p.sum(-1,keepdims=True);assert np.isfinite(p).all() and p.shape==(256,T)
  masks=np.stack([pm,literal,tm,remap]);score=np.stack([p[:,m].sum(-1) for m in masks]+[z['scores'].astype(np.float64)]);q=score.argmax(-1);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS),scores=score,queries=q,boxes=z['boxes'][q],source_boxes=z['boxes'],actual_offsets=offset,positive_masks=masks,shared_CPU64_probabilities=p,stored_native_current=z['scores']);err=float(np.abs(score[0]-z['scores']).max());bridge.append({'split':membership[r['id']],'platform':r['platform'],'maxerror':err,'argmax_equal':bool(q[0]==q[4]),'current_top_score_tie_count':int((score[0]==score[0].max()).sum()),'native_top_score_tie_count':int((z['scores']==z['scores'].max()).sum())})
  dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'id_PRIVATE':r['id'],'split':membership[r['id']],'platform':r['platform'],'proxy_span_actual':proxy,'parser':raw,'parser_fallback_to_current':parser_fallback,'official':official,'official_literal_fallback_independent':lf,'official_remap_fallback_independent':rf,'positive_token_counts':masks.sum(-1).tolist(),'actual_text_sha256':hashlib.sha256(actual.encode()).hexdigest(),'input_tokenization_unchanged':True,'GTclass_consumed_only_by_official_diagnostic_arms':True,'GTbbox_read':False})
  if (i+1)%32==0:dump(out/'PROGRESS.json',{'completed':i+1,'n':336})
 dump(out/'CLASS_ALLOWLIST_RECEIPTS.json',class_receipts);dump(out/'NATIVE_BRIDGE_PRIVATE.json',bridge);seal(out,'R48_ALL5X336_SOURCE_READOUT_PREDICTIONS_SEALED_BEFORE_GTBOX_SCORE',n=336,arms=ARMS,parser_seal_sha256=sha(parsed/'SEALED.json'),source_cache_seal_sha256=sha(source/'SEALED.json'),source_parameters_features_boxes_unchanged=True,no_GPU_grounder_reencoding_or_TTA=True,no_GTbox_or_FINAL_read=True,seconds=time.perf_counter()-start);print('R48_FIVE_ARMS_SEALED',flush=True)
if __name__=='__main__':main()
