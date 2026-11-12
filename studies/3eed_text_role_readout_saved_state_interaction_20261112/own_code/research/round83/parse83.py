"""Reuse frozen local Stanza and tokenizer; no vision models, GT, or downloads."""
from pathlib import Path
import sys,time,re,collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round76'));from common76 import *
from roles83 import roles,spanmask
sys.path.insert(0,str(ROOT/'crossplatform/round48'));from parser48 import parse_subject,dependency_document
H=ROOT/'research/round83';B=ROOT/'crossplatform/round83';p=read(H/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H/'experiment/PARSER_CODE_LOCK.json'))
sys.path.insert(0,str(ROOT/'crossplatform/round20/runtime'));import stanza,torch
from transformers import RobertaTokenizerFast
torch.set_num_threads(4);asset=read(ROOT/'crossplatform/round20/assets/STANZA_ASSET_RECEIPT.json');nlp=stanza.Pipeline('en',dir=str(ROOT/'crossplatform/round20/assets/stanza'),processors='tokenize,mwt,pos,lemma,depparse',package=None,use_gpu=False,download_method=None,verbose=False,**asset['kwargs']);tokenizer=RobertaTokenizerFast.from_pretrained(str(ROOT/'crossplatform/upstream/data/roberta_base'),local_files_only=True)
O=B/'roles2304_attempt001';O.mkdir(parents=True);(O/'cases_PRIVATE').mkdir();counts=collections.Counter();start=time.perf_counter()
for i,r in enumerate(read(ROOT/'research/round69/input/ALL2304_PRIVATE.json')):
 text=r['caption']+' . not mentioned';tok=tokenizer(text,return_offsets_mapping=True);off=np.array(tok['offset_mapping']);document=dependency_document(nlp(r['caption']));subject=parse_subject(r['caption'],document)
 with np.load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:native=z['A_token_mask'].copy();L=z['A_lang0'].shape[1]
 assert len(off)==len(native)==L;assert tokenizer.convert_ids_to_tokens(tok['input_ids'][-1])==tokenizer.eos_token;assert off[-2,1]==len(text) and text[off[-2,0]:off[-2,1]].strip()=='mentioned'
 result=roles(r['caption'],document,subject,off,native,r['id']);record={'caption_sha256':hashlib.sha256(r['caption'].encode()).hexdigest(),'input_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'document':document,'subject':subject,'offsets':off.tolist(),**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items()}};dump(O/'cases_PRIVATE'/(r['id']+'.json'),record);counts[result['reason']]+=1;counts['usable']+=int(result['usable']);counts['attribute_enabled']+=int(result['usable'] and result['attribute_count']>0)
 if(i+1)%128==0:print('PARSE83',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert not torch.cuda.is_initialized();dump(O/'COUNTS.json',{'n':2304,'counts':dict(counts),'seconds':time.perf_counter()-start,'GT_read':False,'new_models_downloaded':False,'existing_CPU_text_parser_only':True});seal(O,'R83_CAPTION_ONLY_ROLE_MASKS_FROZEN_BEFORE_MODEL_READOUT_AND_LABELS',n=2304);print(read(O/'COUNTS.json'))
