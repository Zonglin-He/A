"""GT-free component readout and expanded independent-instance evidence pool."""
import sys,re,numpy as np
from common27 import *
from observations27 import load
sys.path[:0]=[str(ROOT/'crossplatform/round17'),str(ROOT/'crossplatform/round7')]
from pair_geometry import prepare
from regression_adapter import LEXICON
from prepareZ import choose as old_choose
def softmax(x):
    x=x-x.max(-1,keepdims=True);p=np.exp(x);return p/p.sum(-1,keepdims=True)
def masks(r,g,tokenizer,z):
    text=r['caption']+' . not mentioned';tok=tokenizer([text],return_offsets_mapping=True);off=np.array(tok['offset_mapping'][0]);assert len(off)==len(z['tokens']);base=g['base_text'];end=base.get('subject',{}).get('span',[0,len(r['caption'])])[1];match=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',r['caption'][:end],re.I)
    main=np.array(z['token_mask'],bool)
    if match:main=(off[:,1]>match.start())&(off[:,0]<match.end())
    def spanmask(ranges):
        a=np.zeros(len(off),bool)
        for lo,hi in ranges:a|=(off[:,1]>lo)&(off[:,0]<hi)&(off[:,1]>off[:,0])
        return a
    own=base.get('explicit_appearance_field')
    ranges=[own['span']] if own else [base['subject']['span']];attr=spanmask(ranges)&~main
    # Punctuation/function words are not independent semantic attributes.
    for i,(lo,hi) in enumerate(off):
        if not re.search(r'[A-Za-z0-9]',text[lo:hi]) or text[lo:hi].strip().lower() in {'a','an','the','is','it','this','object','type','and','with','of','to'}:attr[i]=False
    aux=spanmask([x['span'] for x in g['references']]);rel=spanmask([(m.start(),m.end()) for m in re.finditer(r'\b(left|right|behind|front|near|beside|closest|nearest|between|center|centre|middle)\b',r['caption'],re.I)]);pron=spanmask([(m.start(),m.end()) for m in re.finditer(r'\b(it|its|they|their|them)\b',r['caption'][:end],re.I)])
    return dict(main=main,attr=attr,pron=pron,rel=rel,aux=aux),('person' if match and match.group().lower() in ['pedestrian','person','people','man','woman','child','boy','girl','walker','worker'] else match.group().lower() if match else g['category'].lower())
def main():
    from transformers import RobertaTokenizerFast
    lock('PREPARE_LOCK.json');verify(BASE/'sourceQ64_attempt001');out=BASE/'preparedQ64_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();(out/'arrays_PRIVATE').mkdir();tokenizer=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True);stats={};maxerr=0.
    for i,r in enumerate(read(EXP.parent/'input/Q64_PRIVATE.json')):
        z=feature(r);g=read(BASE/'textQ64_attempt001/cases_PRIVATE'/(r['id']+'.json'));mask,category=masks(r,g,tokenizer,z);logits=np.asarray(z['query'],float)@np.asarray(z['tokens'],float).T/.07;p=softmax(logits);err=float(np.max(np.abs(p@np.asarray(z['token_mask'],bool)-z['scores'])));maxerr=max(maxerr,err);assert err<2e-5
        component=np.stack([p@mask[k] for k in ['main','attr','pron','rel','aux']],1);delta=component[:,1:4].sum(1)-component[:,4];delta=(delta-np.median(delta))/max(float(np.std(delta)),1e-6);delta=np.clip(delta,-2,2)
        frame,scene,binding=load(r);raw=prepare(scene,z['boxes'],r['id']);legacy=old_choose(scene,z,raw);support=np.asarray(raw['point_support_count']);coverage=np.asarray(raw['point_support_fraction']);pi=np.asarray(raw['projection_iou']);valid=(support>=3)&(coverage>=.15)&(pi>=.1)&(z['boxes'][:,3:]>0).all(1)[:,None];quality=np.where(valid,coverage*np.sqrt(np.maximum(pi,0)),0.);q0=int(z['scores'].argmax());M=len(scene['objects']);classset={0} if category=='person' else {1} if category in ['bicycle','bike','cyclist','biker'] else {3} if category=='motorcycle' else {5} if category=='bus' else {7} if category in ['truck','trailer'] else {2,7} if category in ['car','vehicle','sedan','van','taxi','suv','jeep'] else set()
        target=[m for m in range(M) if valid[:,m].any() and (not classset or scene['objects'][m]['predicted_class_id'] in classset)];target=sorted(target,key=lambda m:(-float(np.max(z['scores']*quality[:,m])),m))[:6];source_mask=int(np.argmax(quality[q0])) if M and quality[q0].max()>0 else None;pool=list(target)
        if source_mask is not None:pool.append(source_mask)
        for pair in legacy['pairs']:pool+=pair['mask_indices_PRIVATE']
        pool=sorted(set(pool));weights=quality[:,pool].copy() if pool else np.zeros((256,0));weights/=np.maximum(weights.sum(1,keepdims=True),1e-12);np.savez_compressed(out/'arrays_PRIVATE'/(r['id']+'.npz'),component=component,token_delta=delta,weights=weights,source_scores=z['scores'],source_boxes=z['boxes']);payload={'alias':r['alias'],'stage':r['stage'],'platform':r['platform'],'category':category,'mask_pool_PRIVATE':pool,'source_mask_PRIVATE':source_mask,'legacy_preparation_PRIVATE':legacy,'bindings':binding,'component_token_counts':{k:int(v.sum()) for k,v in mask.items()},'objects':len(pool),'source_BBF_replay_max_abs_error':err,'GT_used':False};dump(out/'cases_PRIVATE'/(r['id']+'.json'),payload);key='M' if r['stage']=='M' else 'D';stats.setdefault(key,{'n':0,'candidate_masks':0,'has_pool':0});stats[key]['n']+=1;stats[key]['candidate_masks']+=len(pool);stats[key]['has_pool']+=bool(pool)
        if (i+1)%16==0:print('PREPARE23',i+1,flush=True)
    dump(out/'COUNTS.json',stats);seal(out,'R23_expanded_instance_components_GT_free',n=64,counts=stats,BBF_replay_max_error=maxerr);print('PREPARE23_SEALED',stats,maxerr)
if __name__=='__main__':main()
