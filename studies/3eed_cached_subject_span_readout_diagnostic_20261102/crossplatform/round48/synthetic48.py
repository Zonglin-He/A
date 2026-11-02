"""Pure synthetic parser checks; no datasets, models, GPU or annotations."""
from parser48 import parse_subject
def doc(text,items):
 words=[];cursor=0
 for i,(word,pos,head,rel,lemma) in enumerate(items,1):
  lo=text.index(word,cursor);hi=lo+len(word);cursor=hi;words.append({'id':i,'text':word,'lemma':lemma,'upos':pos,'head':head,'deprel':rel,'start_char':lo,'end_char':hi})
 return {'sentences':[{'words':words}]}
def main():
 tests=[]
 text='Near the car stands a man.';d=doc(text,[('Near','ADP',3,'case','near'),('the','DET',3,'det','the'),('car','NOUN',4,'obl','car'),('stands','VERB',0,'root','stand'),('a','DET',6,'det','a'),('man','NOUN',4,'nsubj','man')]);r=parse_subject(text,d);assert r['head_text']=='man';tests.append('matrixsubject_after_reference_not_firstnoun')
 text='This is a pedestrian.';d=doc(text,[('This','PRON',4,'nsubj','this'),('is','AUX',4,'cop','be'),('a','DET',4,'det','a'),('pedestrian','NOUN',0,'root','pedestrian')]);assert parse_subject(text,d)['head_text']=='pedestrian';tests.append('deictic_copular_root')
 text='Find the man beside the truck.';d=doc(text,[('Find','VERB',0,'root','find'),('the','DET',3,'det','the'),('man','NOUN',1,'obj','man'),('beside','ADP',6,'case','beside'),('the','DET',6,'det','the'),('truck','NOUN',3,'nmod','truck')]);assert parse_subject(text,d)['head_text']=='man';tests.append('imperative_unique_obj')
 text='The man and woman stand.';d=doc(text,[('The','DET',2,'det','the'),('man','NOUN',5,'nsubj','man'),('and','CCONJ',4,'cc','and'),('woman','NOUN',2,'conj','woman'),('stand','VERB',0,'root','stand')]);assert parse_subject(text,d)['status']=='UNKNOWN';tests.append('coordinated_subject_unknown')
 text='Object type: traffic cone;';d=doc(text,[('Object','NOUN',0,'root','object'),('type','NOUN',1,'compound','type'),('traffic','NOUN',4,'compound','traffic'),('cone','NOUN',1,'appos','cone')]);r=parse_subject(text,d);assert r['head_text']=='cone' and r['compound_spans']==[[13,20]] and r['head_span']==[21,25];tests.append('explicit_type_head_only_compound_diagnostic')
 d['sentences'][0]['words'][3]['start_char']=None;assert parse_subject(text,d)['status']=='UNKNOWN';tests.append('missing_head_offset_unknown')
 print({'state':'PASS_PURE_SYNTHETIC_PARSER_CHECKS','checks':tests,'actual_data_model_or_GT_access':False})
if __name__=='__main__':main()
