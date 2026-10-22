"""HC64 single-event interval study; all visual/space/full assets stay frozen."""
import os,sys,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(Path(__file__).parent))
D=R/'outputs/joint_event_hc64_v1';P=R/'results/joint_event_hc64_v1'
OLD=R/'outputs/event_query_hc64_v1';B=R.parent/'outputs/tube_temporal_hc64'
ARMS=['NativeG6','JointGuard','Joint','IndependentGuard','RawJointGuard']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.item())+'\n');os.replace(t,p)
def records():return [r for r in map(json.loads,(R/'data/evaluation/development.jsonl').read_text().splitlines())if r['dataset']=='hc2'][:64]
def verify():
    p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
    assert sha(R/'outputs/tastvg_paper_v1/plan.json')==p['paper_plan_sha256']
    for k,v in p['code'].items():assert sha(R/k)==v,k
    for k,v in p['paper_code'].items():assert sha(R/k)==v,k
    for k,v in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/k)==v,k
    return p
def prepare():
    assert not(D/'plan.json').exists();old=read(OLD/'plan.json');seal=read(OLD/'sealed.json');sem=read(OLD/'semantics/sealed.json');paper=read(R/'outputs/tastvg_paper_v1/plan.json');bindings={}
    for row in records():
        k=row['id'];e=seal['files'][k];s=sem['files'][k]
        assert sha(B/'captures'/f'{k}.pt')==old['captures'][k]['capture_sha256']
        assert sha(OLD/e['path'])==e['sha256']and sha(OLD/e['trace_path'])==e['trace_sha256']and sha(OLD/s['path'])==s['sha256']
        bindings[k]=dict(prior=e,semantic=s,capture_sha256=old['captures'][k]['capture_sha256'])
    code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
    for f in ['research/boundary_feedback_v1/core.py','research/boundary_feedback_v1/score.py','research/event_query_v1/legacy_event.py']:code[f]=sha(R/f)
    tokenizer=R/'runtime/referdino/bert-from-frozen-teacher'
    tokenfiles={str(f.relative_to(R)):sha(f)for f in tokenizer.iterdir()if f.name!='model.safetensors'}
    p=dict(schema='joint_event_hc64_v1',primary='JointGuard',arms=ARMS,queries=64,exposure='same repeatedly exposed HC64 development, not independent',
        authorization='User: jointly identify described event with both boundaries; real HC64 test, more improvements/fewer harms; spatial unchanged/full paused',
        fixed=dict(semantics='main-clause lexical predicates excluding noun-attached acl; existing selected whole-track token log-softmax over15 queries relative to subject-noun ownership',
            projection='per-query per-offset normalized six-layer hidden concat -> predicate margin; linear ridge1, full8 and8 leave-one fits; train-only label scaling',
            mask='sigmoid projected margins with intercept matching native posterior expected occupancy, no GT duration',
            pair='mean log soft event intersection/union for whole interval across all predicates; no assumed order between clauses',
            posterior='native q0 times exp(joint score), preserving each eight-anchor membership class mass, original physical MBR',
            guard='both-offset aggregate LOO prediction gain>0; all18 full/LOO semantic views favor proposed interval; no predicate loses combined softIoU; otherwise exact native fallback',
            comparisons='Joint removes guard; IndependentGuard uses same semantics+guard with independent endpoint rise/fall score; RawJointGuard uses old raw event curve with same projection/completeness/guard'),
        new_visual_encoder=0,new_decoder=0,new_expert_calls=0,query_updates=0,new_backbone_training=False,
        lightweight_fit='temporary closed-form pseudo-semantic projection only; no retained cross-query state',
        CPU_text_parser='same existing Stanza query parser; no visual/expert model forward',
        maximum_CPU_seconds=600,bindings=bindings,code=code,tokenizer_files=tokenfiles,paper_code=paper['code'],provider_sources=paper['provider_sources'],
        paper_plan_sha256=sha(R/'outputs/tastvg_paper_v1/plan.json'),labels_sha256=old['labels_sha256'],spatial='original complete-box geometry/parameters/DP unchanged; normal per-interval path and fixed-native-space metrics',
        GT_in_inference=False,all64_seal_before_new_GT=True,full_status='paused_by_user',automatic_resume=False,
        references=['https://minghangz.github.io/uploads/CPL/CPL_paper.pdf','https://github.com/minghangz/cpl/blob/main/models/cpl.py'],
        literature_scope='whole-proposal semantic completeness and confusing same-video contexts; no claim to reproduce trained CPL or to have its masked-word decoder')
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p)
    put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['bindings','paper_code','provider_sources']})
    for k in code:
        f=D/'source_snapshot'/k;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((R/k).read_bytes())
    put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def text_groups(caption,parser,tokenizer,old_groups):
    from research.event_query_v1.legacy_event import event_words,word_tokens
    doc=parser.nlp(caption);words=[]
    for sentence in doc.sentences:
        for token in sentence.tokens:
            for w in token.words:words.append(dict(id=w.id,head=w.head,upos=w.upos,deprel=w.deprel,text=w.text,lemma=w.lemma,start=token.start_char,end=token.end_char))
    text=' '.join(caption.lower().split());offsets=tokenizer([text+'.'if not text.endswith('.')else text],return_offsets_mapping=True)['offset_mapping'][0]
    check,_=word_tokens(caption,event_words(words),offsets);assert check==old_groups,'Original event token mapping must reproduce exactly'
    roots=[w for w in words if w['head']==0 and w['upos']=='VERB'];chosen={w['id']for w in roots}
    if not chosen:
        chosen={w['id']for w in words if w['upos']=='VERB'and w['deprel'].split(':')[0]=='acl'}
    for _ in words:
        more={w['id']for w in words if w['upos']=='VERB'and w['head']in chosen and w['deprel'].split(':')[0]in {'conj','xcomp','advcl','ccomp'}}
        if more<=chosen:break
        chosen|=more
    verbs=[w for w in words if w['id']in chosen]
    wh=next((w for w in words if w['lemma'].lower()in {'which','what'}and w['deprel']=='det'),None)
    noun=next((w for w in words if wh and w['id']==wh['head']),None)
    if noun is None:noun=next((w for w in words if w['lemma'].lower()in {'who','whom'}),None)
    if noun is None:noun=next((w for w in words if w['deprel'].startswith('nsubj')and w['upos']in {'NOUN','PROPN'}),None)
    if noun is None:noun=next((w for w in words if w['upos']in {'NOUN','PROPN'}),None)
    vg,vr=word_tokens(caption,verbs,offsets);ng,nr=word_tokens(caption,[noun]if noun else [],offsets)
    return vg,ng,dict(predicate_count=len(vg),subject_available=bool(ng),verb_mapping=vr,noun_mapping=nr,all_lexical_verbs=sum(w['upos']=='VERB'for w in words),token_mapping_exact=True)
def predict():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch,numpy as np
    from transformers import AutoTokenizer
    from stvg.query import QuerySubjectParser
    from stvg.runtime import guard as annotation_guard
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.temporal import final_mbr
    from research.boundary_feedback_v1.core import groups_for,reweight
    from research.joint_event_v1.core import relative_event,projections,physical_membership,calibrated_masks,interval_scores,independent_scores,guard
    p=verify();torch.set_num_threads(1);sys.addaudithook(annotation_guard);assert not(D/'sealed.json').exists()and not torch.cuda.is_initialized()
    for f,h in p['tokenizer_files'].items():assert sha(R/f)==h
    tokenizer=AutoTokenizer.from_pretrained(str(R/'runtime/referdino/bert-from-frozen-teacher'),local_files_only=True,use_fast=True)
    parser=QuerySubjectParser(R/'checkpoints/stanza');files={};start=time.perf_counter();put(D/'state.json',dict(status='predicting_CPU',fingerprint=p['fingerprint']))
    for row in records():
        k=row['id'];b=p['bindings'][k];e=b['prior'];s=b['semantic']
        assert sha(B/'captures'/f'{k}.pt')==b['capture_sha256']and sha(OLD/s['path'])==s['sha256']and sha(OLD/e['trace_path'])==e['trace_sha256']and sha(OLD/e['path'])==e['sha256']
        cap=torch.load(B/'captures'/f'{k}.pt',map_location='cpu',weights_only=False);tr=torch.load(OLD/e['trace_path'],map_location='cpu',weights_only=False);sem=torch.load(OLD/s['path'],map_location='cpu',weights_only=False)
        ids=cap['frame_ids'];pos=cap['planned_positions'];pairs=tr['feedback']['pairs'];q0=tr['q']['NativeG6'];patterns,group=groups_for(ids,pos,pairs)
        tick=time.perf_counter();vg,ng,tm=text_groups(row['metadata']['caption'],parser,tokenizer,sem['groups'])
        overlap,width,lengths=physical_membership(ids,pairs);occupancy=float(q0[pairs[:,0],pairs[:,1]]@lengths/width.sum());assert 0<occupancy<1
        base=final_mbr(ids,q0);base_idx=int(np.flatnonzero((pairs==base['indices']).all(1))[0]);qs={'NativeG6':q0};iv={'NativeG6':base['interval']};details={};saved={}
        for kind in ['Relative','Raw']:
            labels=relative_event(sem['premax'].numpy(),vg,ng,sem['selected_query'])if kind=='Relative'and vg and ng else np.asarray(sem['curve'])[:,None]if kind=='Raw'and sem['curve']is not None else None
            supported=labels is not None and np.isfinite(labels).all()
            if supported:
                active=np.std(labels,axis=0)>1e-9;labels=labels[:,active];supported=labels.shape[1]>0
            if not supported:
                names=['Joint','JointGuard','IndependentGuard']if kind=='Relative'else ['RawJointGuard']
                for a in names:qs[a]=q0;iv[a]=base['interval'];details[a]=dict(accept=False,reason='no_valid_semantic_contrast')
                continue
            models,skills=projections(cap['temporal_cache']['offsets'],ids,pos,labels);masks=calibrated_masks(models,width,occupancy)
            combined=(masks[0][0]+masks[1][0])/2;score,logs=interval_scores(combined,overlap,width,lengths)
            view_scores=[interval_scores(v,overlap,width,lengths)[0]for view in masks for v in view]
            names=['Joint','IndependentGuard']if kind=='Relative'else ['RawJointGuard']
            for name in names:
                unary=independent_scores(combined,pairs)if name=='IndependentGuard'else score
                q=reweight(q0,pairs,unary,group);decision=final_mbr(ids,q);idx=int(np.flatnonzero((pairs==decision['indices']).all(1))[0]);accept,diag=guard(idx,base_idx,view_scores,logs,skills)
                if name=='Joint':
                    qs['Joint']=q;iv['Joint']=decision['interval'];details['Joint']=dict(guard_would_accept=accept,**diag)
                    qs['JointGuard']=q if accept else q0;iv['JointGuard']=decision['interval']if accept else base['interval'];details['JointGuard']=diag
                else:qs[name]=q if accept else q0;iv[name]=decision['interval']if accept else base['interval'];details[name]=diag
            saved[kind]=dict(labels=labels,masks=masks,combined=combined,score=score,skills=skills)
        feedback_seconds=time.perf_counter()-tick;geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={}
        for a in ARMS:arrays[a]=evidence_dp(geom.for_interval(iv[a]))['final'].numpy()
        with np.load(OLD/e['path'],allow_pickle=False)as old:
            assert np.array_equal(arrays['NativeG6'],old['NativeG6'])and np.array_equal(iv['NativeG6'],old['NativeG6_interval'])
            arrays['Source']=old['Source'].copy();iv['Source']=old['Source_interval'].tolist()
        out=D/'predictions'/f'{k}.npz';assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=pos)
        trace=D/'traces'/f'{k}.pt';trace.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(q=qs,semantic=saved,details=details,text_metadata=tm,occupancy=occupancy,pairs=pairs,groups=group,patterns=patterns,GT_read=False),trace)
        files[k]=dict(path=str(out.relative_to(D)),sha256=sha(out),trace_path=str(trace.relative_to(D)),trace_sha256=sha(trace),feedback_seconds=feedback_seconds,seconds=time.perf_counter()-tick)
        put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start));print(k,{a:bool(details[a].get('accept'))for a in ['JointGuard','IndependentGuard','RawJointGuard']},flush=True)
        assert time.perf_counter()-start<p['maximum_CPU_seconds']
    verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-start))
    put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
if __name__=='__main__':globals()[sys.argv[1]]()
