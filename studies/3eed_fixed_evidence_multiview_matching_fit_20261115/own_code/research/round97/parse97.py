"""Reuse the existing text-only dependency parser and role inventory, no task labels."""
from pathlib import Path
import sys,json,time,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round97';B=R/'crossplatform/round97'
sys.path[:0]=[str(R/'crossplatform/round48'),str(R/'research/round85'),str(R/'research/round86'),str(R/'research/round87'),str(R/'crossplatform/round20/runtime')]
from parser48 import parse_subject,dependency_document
from evidence85 import parse85
from reference_roles87 import repair_inventory87
import stanza
def main():
 import torch
 torch.set_num_threads(4);out=B/'roles_PRIVATE';out.mkdir(exist_ok=True);asset=json.loads((R/'crossplatform/round20/assets/STANZA_ASSET_RECEIPT.json').read_text());nlp=stanza.Pipeline('en',dir=str(R/'crossplatform/round20/assets/stanza'),processors='tokenize,mwt,pos,lemma,depparse',package=None,use_gpu=False,download_method=None,verbose=False,**asset['kwargs']);rows={r['id']:r for p in sorted((H/'input').glob('*.json')) for r in json.loads(p.read_text())};start=time.perf_counter()
 for n,r in enumerate(rows.values()):
  f=out/(r['id']+'.json')
  if f.exists():assert json.loads(f.read_text())['caption_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest();continue
  doc=dependency_document(nlp(r['caption']));subject=parse_subject(r['caption'],doc);graph=repair_inventory87(r['caption'],doc,parse85(r['caption'],doc,subject));f.write_text(json.dumps({'graph':graph,'doc':doc,'caption_sha256':hashlib.sha256(r['caption'].encode()).hexdigest()},ensure_ascii=False)+'\n')
  if (n+1)%256==0:print('PARSE97',n+1,len(rows),round(time.perf_counter()-start,1),flush=True)
 (B/'ROLE_PARSE_COMPLETE.json').write_text(json.dumps({'n':len(rows),'seconds':time.perf_counter()-start,'task_labels_read':False,'new_models_downloaded':False})+'\n');print('PARSE97_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
