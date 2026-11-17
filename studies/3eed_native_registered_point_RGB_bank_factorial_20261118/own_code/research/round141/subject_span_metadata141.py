"""New-cohort text-only P0 eligibility check; no model, task label or score."""
from common141 import *
import numpy as np,collections
sys.path.insert(0,str(R/'crossplatform/round48'));from parser48 import parse_subject
from transformers import RobertaTokenizerFast
def main():
 check();tok=RobertaTokenizerFast.from_pretrained(str(R/'crossplatform/upstream/data/roberta_base'),local_files_only=True);rows=[];counts=collections.Counter();unique={}
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];cap=e['row']['caption'];doc=rd(R/'crossplatform/round97/roles_PRIVATE'/(sid+'.json'));assert doc['caption_sha256']==hashlib.sha256(cap.encode()).hexdigest();p=parse_subject(cap,doc['doc']);text=cap+' . not mentioned';offset=np.asarray(tok(text,return_offsets_mapping=True)['offset_mapping']);z=np.load(R/'crossplatform/round131/formal'/(src+'_to_'+tgt)/'prefix_PRIVATE'/(sid+'.npz'));current=z['token_mask'];assert len(offset)==len(current);mask=current.copy()
  if p['status']=='FOUND':
   lo,hi=p['head_span'];mask=(offset[:,1]>offset[:,0])&(offset[:,0]<hi)&(offset[:,1]>lo);assert mask.any() and (~mask).any()
  changed=not np.array_equal(mask,current);r={'source':src,'target':tgt,'id_PRIVATE':sid,'status':p['status'],'mask_changed':changed,'current_positive_tokens':int(current.sum()),'parser_positive_tokens':int(mask.sum())};rows.append(r);counts['n']+=1;counts[p['status']]+=1;counts['changed']+=changed
  if sid in unique:assert unique[sid]['mask_changed']==changed and unique[sid]['status']==p['status']
  else:unique[sid]=r
 out={'status':'TEXT_ONLY_P0_ELIGIBILITY_NO_TASK_SCORE','directed_inputs':dict(counts),'unique_expressions':len(unique),'unique_mask_changes':sum(r['mask_changed'] for r in unique.values()),'unique_FOUND':sum(r['status']=='FOUND' for r in unique.values()),'unique_UNKNOWN':sum(r['status']!='FOUND' for r in unique.values()),'new_cohort_distinct_from_old_DEV336_diagnostic':True,'no_grounding_Roberta_or_parser_model_inference':True,'uses_existing_caption_dependency_cache_only':True,'no_visual_semantic_identity_claim':True,'GT_or_task_scores_read':False,'R141_predictor_unchanged':True,'FINAL_access':False};dump(H/'SUBJECT_SPAN_METADATA_ROWS_PRIVATE.json',rows);dump(H/'SUBJECT_SPAN_METADATA_ONLY.json',out);print(out,flush=True)
if __name__=='__main__':main()
