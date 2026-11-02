"""Caption-only CPU dependency parsing; sealed before official classes are read."""
import sys,time
from common48 import *
from parser48 import parse_subject,dependency_document
def main():
 a=args();plan,rows,membership,auth=context(a);out=ROOT/plan['parse_output'];out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();(out/'dependencies_PRIVATE').mkdir()
 assert plan['parser_rules_sha256']==sha(ROOT/'research/round48/experiment/PARSER_RULES_LOCK.json')
 # Existing local frozen Stanza assets, same processors as textZ; no download,
 # grounding/VLM model, GPU or metadata annotation access in this phase.
 sys.path.insert(0,str(ROOT/'crossplatform/round20/runtime'));import stanza,torch
 torch.set_num_threads(4);assert not torch.cuda.is_initialized()
 receipt=read(ROOT/'crossplatform/round20/assets/STANZA_ASSET_RECEIPT.json');assert sha(ROOT/'crossplatform/round20/assets/STANZA_ASSET_RECEIPT.json')==plan['Stanza_receipt_sha256']
 nlp=stanza.Pipeline('en',dir=str(ROOT/'crossplatform/round20/assets/stanza'),processors='tokenize,mwt,pos,lemma,depparse',package=None,use_gpu=False,download_method=None,verbose=False,**receipt['kwargs']);start=time.perf_counter();counts={}
 for i,r in enumerate(rows):
  dep=dependency_document(nlp(r['caption']));result=parse_subject(r['caption'],dep);dump(out/'dependencies_PRIVATE'/(r['id']+'.json'),dep);dump(out/'cases_PRIVATE'/(r['id']+'.json'),dict(result,id_PRIVATE=r['id'],split=membership[r['id']],platform=r['platform'],caption_sha256=hashlib.sha256(r['caption'].encode()).hexdigest(),parser_uses_metadata_class_or_boxes=False));key=membership[r['id']]+'/'+r['platform'];counts.setdefault(key,{'n':0,'FOUND':0,'UNKNOWN':0,'explicit':0});counts[key]['n']+=1;counts[key][result['status']]+=1;counts[key]['explicit']+=int(result['structured'])
  if (i+1)%32==0:dump(out/'PROGRESS.json',{'completed':i+1,'n':336})
 dump(out/'COUNTS.json',counts);seal(out,'R48_CAPTION_ONLY_PARSER_ALL336_SEALED_BEFORE_METADATA_CLASS_READ',n=336,no_metadata_class_bbox_GT_or_image_point_data_read=True,CPU_existing_Stanza_only=True,program_sha256=sha(__file__),rules_sha256=plan['parser_rules_sha256'],seconds=time.perf_counter()-start);print('R48_PARSER_SEALED',counts,flush=True)
if __name__=='__main__':main()
