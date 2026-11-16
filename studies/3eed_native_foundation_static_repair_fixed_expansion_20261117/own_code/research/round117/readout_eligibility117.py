"""No-task-GT caption-only readout eligibility on new cohorts; frozen R48 parser."""
from pathlib import Path
import sys,json,hashlib,re,numpy as np,torch
from transformers import RobertaTokenizerFast
R=Path(__file__).resolve().parents[2];H=R/'research/round117';rd=lambda p:json.loads(Path(p).read_text());sys.path.insert(0,str(R/'crossplatform/round48'));from parser48 import parse_subject
sys.path.insert(0,str(R/'crossplatform/round7'));from regression_adapter import first_noun_mask

def main():
 tok=RobertaTokenizerFast.from_pretrained(str(R/'crossplatform/round3/p0/upstream/data/roberta_base'),local_files_only=True);out=[];summary=[]
 for cohort,rrt,B in [('DEV',R/'research/round99/input',R/'crossplatform/round111'),('EXP',R/'research/round109/input',R/'crossplatform/round117/formal')]:
  for target in ['waymo','drone','quad']:
   rows=rd(rrt/(f'TARGET_{target.upper()}'+('128' if cohort=='DEV' else '256')+'_PRIVATE.json'));n=unknown=found=diff=aliases=0
   for r in rows:
    candidates=[B/(src+'_to_'+target)/'prefix_PRIVATE'/(r['id']+'.npz') for src in ['waymo','drone','quad'] if src!=target];paths=[f for f in candidates if f.exists()]
    if not paths:aliases+=1;continue
    with np.load(paths[0]) as z:old=z['token_mask'].astype(bool)
    for f in paths[1:]:
     with np.load(f) as z:assert np.array_equal(old,z['token_mask'])
    doc=rd(R/'crossplatform/round97/roles_PRIVATE'/(r['id']+'.json'));assert doc['caption_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest();text=r['caption']+' . not mentioned';enc=tok(text,return_offsets_mapping=True);offset=np.array(enc['offset_mapping']);assert len(old)==len(offset)
    try:lex=first_noun_mask(text,offset).numpy();fallback=False
    except ValueError:lex=offset[:,1]>offset[:,0];fallback=True
    assert np.array_equal(old,lex),'Current tokenizer/legacy mask parityfailed';parsed=parse_subject(r['caption'],doc['doc']);mask=old.copy()
    if parsed['status']=='FOUND':
     found+=1;lo,hi=parsed['head_span'];mask=(offset[:,1]>lo)&(offset[:,0]<hi);assert mask.any()
    else:unknown+=1
    change=not np.array_equal(old,mask);diff+=change;n+=1;out.append({'cohort':cohort,'target':target,'id_PRIVATE':r['id'],'caption_PRIVATE':r['caption'],'parser':parsed,'mask_changes':change,'legacy_fallback':fallback,'old_token_mask_PRIVATE':old.tolist(),'parser_token_mask_PRIVATE':mask.tolist(),'GT_used':False})
   summary.append({'cohort':cohort,'target':target,'unique_expressions':n,'protected_alias_fallback_excluded_from_new_parse':aliases,'FOUND':found,'UNKNOWN':unknown,'effective_mask_differences':diff,'GT_used':False})
 (H/'READOUT_ELIGIBILITY_PRIVATE.json').write_text(json.dumps(out,indent=2)+'\n');res={'rows':summary,'existing_R48_parser_unmodified':True,'no_new_grounder_NLP_model_or_taskGT':True,'FINAL_access':False};(H/'READOUT_ELIGIBILITY_SUMMARY.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
