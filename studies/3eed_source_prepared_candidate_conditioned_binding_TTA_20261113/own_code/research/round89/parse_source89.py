"""Source-only caption roles; no target manifest/features/labels imported."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round76'));from common76 import *
sys.path.insert(0,str(ROOT/'crossplatform/round48'));from parser48 import parse_subject,dependency_document
sys.path.insert(0,str(ROOT/'research/round85'));from evidence85 import parse85
sys.path.insert(0,str(ROOT/'research/round86'));sys.path.insert(0,str(ROOT/'research/round87'));from reference_roles87 import repair_inventory87
sys.path.insert(0,str(ROOT/'crossplatform/round20/runtime'));import stanza
H=ROOT/'research/round89';O=ROOT/'crossplatform/round89/source_roles640_attempt001';(O/'cases_PRIVATE').mkdir(parents=True,exist_ok=False);(O/'dependencies_PRIVATE').mkdir();asset=read(ROOT/'crossplatform/round20/assets/STANZA_ASSET_RECEIPT.json');nlp=stanza.Pipeline('en',dir=str(ROOT/'crossplatform/round20/assets/stanza'),processors='tokenize,mwt,pos,lemma,depparse',package=None,use_gpu=False,download_method=None,verbose=False,**asset['kwargs']);rows=source_rows('TRAIN512')+source_rows('VAL128');stats=[];start=time.perf_counter()
for i,r in enumerate(rows):
 assert r['platform']=='waymo';doc=dependency_document(nlp(r['caption']));subject=parse_subject(r['caption'],doc);graph=repair_inventory87(r['caption'],doc,parse85(r['caption'],doc,subject));dump(O/'cases_PRIVATE'/(r['id']+'.json'),{'graph':graph,'caption_sha256':hashlib.sha256(r['caption'].encode()).hexdigest()});dump(O/'dependencies_PRIVATE'/(r['id']+'.json'),doc);stats.append({'row':'SOURCE'+str(i+1).zfill(4),'split':'TRAIN512' if i<512 else 'VAL128','roles':len(graph['entities']),'unknown':not graph['usable']})
 if (i+1)%128==0:print('PARSE89',i+1,flush=True)
dump(O/'COUNTS.json',{'n':640,'seconds':time.perf_counter()-start,'target_inputs_read':False,'GT_read':False,'new_models_downloaded':False});seal(O,'R89_SOURCE640_CAPTION_ROLES_ONLY_SEALED',n=640);dump(H/'SOURCE_ROLE_ROWS.json',stats);print(read(O/'COUNTS.json'))
