"""Check raw source-role offsets against actual normalized source text token order."""
from common140 import *
import numpy as np
from transformers import RobertaTokenizerFast
def main():
 check();tok=RobertaTokenizerFast.from_pretrained(str(R/'crossplatform/upstream/data/roberta_base'),local_files_only=True);reg=rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json');rows=[]
 for e in reg:
  src=e['entry']['source'];cap=e['entry']['row']['caption'];actual=' '.join(cap.replace(',',' ,').split())+' . not mentioned';raw=cap+' . not mentioned';a=tok(raw,return_offsets_mapping=True);b=tok(actual,return_offsets_mapping=True);z=np.load(R/e['entry']['files']['prefix']['path']);assert len(b['input_ids'])==z['lang0'].shape[1];length_equal=len(a['input_ids'])==len(b['input_ids']);diff=[]
  if length_equal:
   for j,(oa,ob) in enumerate(zip(a['offset_mapping'],b['offset_mapping'])):
    aa=raw[oa[0]:oa[1]].strip();bb=actual[ob[0]:ob[1]].strip()
    if aa!=bb:diff.append(j)
  rt=np.load(R/e['entry']['files']['route']['path']);roles=rt['token_roles'];relations=rt['relation_tokens'];touched=sum(bool(roles[:,j].any() or relations[:,j].any()) for j in diff) if length_equal else None;rows.append({'source':src,'id_PRIVATE':e['entry']['row']['id'],'metadata_raw_vs_actual_text_changed':raw!=actual,'token_lengths_equal':length_equal,'input_token_ID_changes':sum(x!=y for x,y in zip(a['input_ids'],b['input_ids'])),'semantic_piece_mismatches':len(diff),'mismatch_assigned_to_role_or_relation':touched})
 out={'status':'SOURCE_TEXT_METADATA_ONLY','rows':len(rows),'normalized_prefix_tokenlength_all_pass':True,'counts_by_source':{src:{'n':sum(r['source']==src for r in rows),'metadata_different':sum(r['source']==src and r['metadata_raw_vs_actual_text_changed'] for r in rows),'tokenlength_mismatch':sum(r['source']==src and not r['token_lengths_equal'] for r in rows),'semantic_piece_mismatches':sum(r['semantic_piece_mismatches'] for r in rows if r['source']==src),'role_relation_mismatches':sum(r['mismatch_assigned_to_role_or_relation'] or 0 for r in rows if r['source']==src)} for src in ['waymo','drone','quad']},'actual_source_capture_normalizes_commas_before_encoding':True,'source_prepared_weights_or_R140_predictions_changed':False,'new_grounding_or_Roberta_model_forwards':0,'target_inputs_or_GT_read':False,'FINAL_access':False};dump(H/'SOURCE_TEXT_METADATA_ROWS_PRIVATE.json',rows);dump(H/'SOURCE_TEXT_METADATA_ONLY.json',out);print(out,flush=True)
if __name__=='__main__':main()
