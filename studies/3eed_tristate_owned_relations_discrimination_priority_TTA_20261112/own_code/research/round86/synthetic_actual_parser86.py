"""Use the existing local text parser on synthetic relative/existential clauses."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round20/runtime'));sys.path.insert(0,str(ROOT/'crossplatform/round48'))
import stanza,torch
from parser48 import dependency_document
from evidence86 import relations86
torch.set_num_threads(4);receipt=json.loads((ROOT/'crossplatform/round20/assets/STANZA_ASSET_RECEIPT.json').read_text());nlp=stanza.Pipeline('en',dir=str(ROOT/'crossplatform/round20/assets/stanza'),processors='tokenize,mwt,pos,lemma,depparse',package=None,use_gpu=False,download_method=None,verbose=False,**receipt['kwargs'])
texts=[('The car left of the truck, which is right of the bus in the image.',['car','truck','bus'],{('T','R0','LEFT'),('R0','R1','RIGHT')}),('The car is in the image. There is a tree on the left and a truck on the right.',['car','tree','truck'],{('R0','T','LEFT'),('R1','T','RIGHT')})];out=[]
for text,names,expected in texts:
 entities=[]
 for i,head in enumerate(names):
  a=text.index(head);entities.append({'role':'T' if i==0 else 'R'+str(i-1),'head':head,'phrase':head,'span':[a,a+len(head)],'attributes':[]})
 doc=dependency_document(nlp(text));edges=relations86(text,doc,{'usable':True,'entities':entities});actual={(e['subject'],e['reference'],e['predicate']) for e in edges};assert actual==expected,(actual,expected);out.append({'expected_edges':len(expected),'actual_edges':len(actual),'spurious_edges':0})
assert not torch.cuda.is_initialized();result={'state':'PASS_R86_EXISTING_STANZA_ACTUAL_SYNTHETIC_RELATIVE_AND_MULTIPLE_EXISTENTIAL_OWNERS','tests':out,'dataset_GT_read':False,'no_models_downloaded':True,'CPU_text_parser_only':True};(Path(__file__).resolve().parent/'ACTUAL_PARSER_SYNTHETIC_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
