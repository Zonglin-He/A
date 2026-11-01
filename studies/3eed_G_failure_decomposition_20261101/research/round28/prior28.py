"""Audit source readout probability space using saved embeddings, no model/GPU."""
import sys,re,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round28'),str(ROOT/'crossplatform/round7')]
from common28 import *
from regression_adapter import LEXICON
from transformers import RobertaTokenizerFast
def main():
 tok=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True);out=[];maxerr=0.
 for r in rows():
  z=feature(r);text=r['caption']+' . not mentioned';offset=np.array(tok([text],return_offsets_mapping=True)['offset_mapping'][0]);m=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',text,re.I);mask=(offset[:,1]>m.start())&(offset[:,0]<m.end()) if m else offset[:,1]>offset[:,0];assert np.array_equal(mask,z['token_mask']);v=np.asarray(z['query'],float)@np.asarray(z['tokens'],float).T/.07;v-=v.max(1,keepdims=True);p=np.exp(v);p/=p.sum(1,keepdims=True);s=p@mask;err=float(abs(s-z['scores']).max());assert err<2e-5;maxerr=max(maxerr,err);out.append({'alias':r['alias'],'platform':r['platform'],'selected_span_is_first_noun':bool(m),'selected_tokens':int(mask.sum()),'all_caption_content_tokens':int(((offset[:,1]>offset[:,0])&(offset[:,0]<len(r['caption']))).sum()),'sum_over256_query_scores':float(z['scores'].sum()),'max_score':float(z['scores'].max())})
 dump(EXP/'SOURCE_PRIOR_AUDIT_ROWS.json',out);dump(EXP/'SOURCE_PRIOR_AUDIT.json',{'n':528,'first_noun_readout_n':sum(r['selected_span_is_first_noun'] for r in out),'cached_token_mask_exactly_reconstructed':True,'BBF_score_replay_max_abs_error':maxerr,'source_softmax_axis':'text token positions within each query, not distribution across256instances','source_full_caption_still_encoded':True,'not_proof_that_source_ignores_attributes_or_relations':True,'candidate_score_sum_min':min(r['sum_over256_query_scores'] for r in out),'candidate_score_sum_max':max(r['sum_over256_query_scores'] for r in out),'Qwen_score':'binary YesNo logodds per RGB candidate, neither classposterior nor normalizedinstanceprobability','no_model_optimizer_GPU_or_new_predictions':True});print('PRIOR28',sum(r['selected_span_is_first_noun'] for r in out),maxerr)
if __name__=='__main__':main()
