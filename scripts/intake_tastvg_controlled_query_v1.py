"""Bounded CPU intake of real query text and Vid referent IDs, never GT times."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,status as atomic_status
from scripts.tastvg_controlled_query_math_v1 import linguistic_signature,norm,query_form,text_hash,choose_donor
BASE=ROOT/'artifacts/tastvg_controlled_query_v1'
PLAN=ROOT/'artifacts/tastvg_current_correction_views_v1'


def run():
    import torch,stanza
    torch.set_num_threads(4);tick=time.time()
    assert not (BASE/'PRIVATE_MATCHES.json').exists()
    panels={d:read(PLAN/d/'PLAN.json')['rows'] for d in ['vidstg','hc2']}
    vidpath=ROOT/'external/VidSTG-Dataset/annotations/test_annotations.json'
    raw=read(vidpath)
    # Annotation-bearing file whitelist: no temporal_gt, relations or trajectories.
    vidmeta=[];ordinal=0
    for a in raw:
        for field in ['captions','questions']:
            for q in a[field]:
                vidmeta.append(dict(key=f'vidstg_test:{ordinal:06}',source=a['vid'],caption=q['description'],
                    target_id=q['target_id'],segment=[a['used_segment']['begin_fid'],a['used_segment']['end_fid']+1],
                    form='question' if field=='questions' else 'declarative'))
                ordinal+=1
    assert ordinal==10303
    del raw
    bykey={x['key']:x for x in vidmeta};sources={r['source'] for r in panels['vidstg']}
    video_by_source={r['source']:r['input']['video_sha256'] for r in panels['vidstg']}
    pool={'vidstg':[{**x,'video_sha256':video_by_source[x['source']]} for x in vidmeta if x['source'] in sources]}
    hcpath=ROOT/'artifacts/tastvg_best_full_v1/hc2/PLAN.json'
    pool['hc2']=[dict(key=r['key'],source=r['source'],caption=r['input']['caption'],target_id=None,
        segment=[0,r['input']['frame_count']],form='declarative',video_sha256=r['input']['video_sha256'])
        for r in read(hcpath)['rows']]
    recipients={}
    for ds,rows in panels.items():
        recipients[ds]=[]
        for i,r in enumerate(rows):
            q=bykey[r['key']] if ds=='vidstg' else dict(target_id=None,segment=[0,r['input']['frame_count']],form='declarative')
            assert ds!='vidstg' or norm(q['caption'])==norm(r['input']['caption'])
            recipients[ds].append(dict(key=r['key'],index=i,caption=r['input']['caption'],source=r['source'],
                video_sha256=r['input']['video_sha256'],target_id=q['target_id'],segment=q['segment'],form=q['form']))
    nlp=stanza.Pipeline('en',dir=str(ROOT/'.cache/stanza'),package=None,
        processors={'tokenize':'ewt','mwt':'ewt','pos':'ewt_nocharlm','lemma':'ewt_nocharlm','depparse':'ewt_nocharlm'},
        use_gpu=False,download_method=None,verbose=False)
    texts=sorted({norm(x['caption']) for xs in list(pool.values())+list(recipients.values()) for x in xs})
    cache={}
    for start in range(0,len(texts),32):
        chunk=texts[start:start+32];docs=nlp.bulk_process(chunk)
        for text,doc in zip(chunk,docs):
            words=[];offset=0
            for sent in doc.sentences:
                words.extend(dict(id=w.id+offset,local_id=w.id,text=w.text,lemma=w.lemma.lower(),upos=w.upos,
                    head=w.head+offset if w.head else 0,local_head=w.head,deprel=w.deprel) for w in sent.words)
                offset+=len(sent.words)
            cache[text]=dict(words=words,signature=linguistic_signature(words))
        atomic_status(BASE/'INTAKE_STATUS.json',dict(status='CPU_text_parsing',done=min(start+32,len(texts)),total=len(texts),GPU=False,time=time.time()))
        if start%256==0:print('CONTROLLED_QUERY_PARSE',start,len(texts),flush=True)
    matches={};summary={}
    for ds in pool:
        for x in pool[ds]+recipients[ds]:x['signature']=cache[norm(x['caption'])]['signature']
        matches[ds]={};counts=collections.Counter()
        for r in recipients[ds]:
            pp={arm:choose_donor(r,pool[ds],arm,ds) for arm in ['event','subject']}
            matches[ds][str(r['index'])]=dict(recipient=r,pairs=pp)
            for arm,p in pp.items():counts[arm+'/'+(p['kind'] if p else 'unavailable')]+=1
            counts['both_available']+=int(all(pp.values()))
        summary[ds]=dict(recipients=48,donor_pool_queries=len(pool[ds]),counts=dict(counts),
            same_exact_media_multicaption_sources=sum(len({p['caption'] for p in pool[ds] if p['video_sha256']==r['video_sha256']})>1 for r in recipients[ds]))
    write(BASE/'PRIVATE_NLP_TEXT.json',cache)
    write(BASE/'PRIVATE_DONOR_POOL.json',pool)
    write(BASE/'PRIVATE_MATCHES.json',matches)
    assets={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'.cache/stanza/en').rglob('*.pt')}
    write(BASE/'INTAKE.json',dict(status='text_only_matching_complete',time=time.time(),seconds=time.time()-tick,
        source_files={str(vidpath.relative_to(ROOT)):sha(vidpath),str(hcpath.relative_to(ROOT)):sha(hcpath)},
        annotation_file_read=True,annotation_fields_used=['vid','used_segment','captions.description','questions.description','target_id'],
        GT_temporal_or_boxes_used_for_matching=False,GT_time_values_saved=False,parser_version=stanza.__version__,
        parser_model_hashes=assets,unique_texts=len(texts),summary=summary))
    print('CONTROLLED_QUERY_INTAKE',json.dumps(summary),flush=True)

if __name__=='__main__':run()
