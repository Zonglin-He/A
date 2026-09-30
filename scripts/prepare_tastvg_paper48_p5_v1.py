"""Lock HC2 from available official validation media, by source hash only."""
import sys,time,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_paper48_p5_common_v1 import BASE,CHECKPOINT,CHECKPOINT_SHA,frame_ids,read,write,sha
from scripts.tastvg_paper48_common_v1 import CONDS

def h(s):return hashlib.sha256(s.encode()).hexdigest()
def run():
    if (BASE/'LOCK.json').exists():
        from scripts.tastvg_paper48_p5_common_v1 import verify
        verify();return
    import torch
    from vg_tta.foreground_runtime import QuerySubjectParser
    torch.set_num_threads(4)
    annotation=ROOT/'data/hcstvg2_confirm512/annotations/valv2_proc.json'
    # Annotation-bearing intake is explicitly disclosed. Whitelist immediately;
    # no label coordinates, spans, predictions or scores enter selection/runtime.
    fields=['original_video_id','frame_count','width','height','video_path','caption','video_id']
    meta=[{k:r[k] for k in fields} for r in read(annotation)]
    groups=collections.defaultdict(list)
    for r in meta:
        source=Path(r['video_path']).stem.split('_',1)[1]
        groups[source].append(r)
    sources=sorted(groups,key=lambda s:(h('P48-HC2-source-v1|'+s),s))[:128]
    assert len(sources)==128
    parser=QuerySubjectParser(ROOT/'.cache/stanza');rows=[];subject_files={}
    for i,source in enumerate(sources):
        r=min(groups[source],key=lambda r:(h('P48-HC2-query-v1|'+r['original_video_id']),r['original_video_id']))
        path=ROOT/'data/hcstvg2_confirm512/video'/r['video_path'];assert path.is_file()
        ids=frame_ids(r['frame_count']);q=dict(index=r['video_id'],source=source,original_video_id=r['original_video_id'],kind='hcstvg',caption=r['caption'].lower(),width=r['width'],height=r['height'],frame_count=r['frame_count'],fps=r['frame_count']/20.,duration=20.,video_path=str(path),video_sha256=sha(path),frame_ids=ids)
        rows.append(dict(ordinal=i,key='paper48_hc2:'+r['original_video_id'],source=source,input=q,frame_ids=ids,query_type='declarative',annotation_index=r['video_id']))
        f=BASE/'subjects'/f'{i:05}.json'
        if not f.exists():write(f,dict(ordinal=i,caption_sha256=h(q['caption']),parses=parser(q['caption'])))
        subject_files[f.name]=sha(f)
    seq=sorted(range(128),key=lambda i:(h('P48-HC2-order-v1|'+rows[i]['source']),rows[i]['source']))
    plan=dict(name='P5',rows=rows,orders={'order1':seq},conditions=CONDS,availability=25,total=768,sources=128,source_checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,same_domain=True,GT_used_for_selection=False,annotation_metadata_intake=True,globally_fresh=False)
    write(BASE/'PLAN.json',plan);needed=seq[::4]
    write(BASE/'EXPERT_PLAN.json',dict(rows=rows,conditions=CONDS,expert_needed=needed,conditions_by_parent={str(i):CONDS for i in needed},total=len(needed)*len(CONDS)))
    write(BASE/'SUBJECT_BARRIER.json',dict(count=128,files=subject_files,GT_read=False))
    write(BASE/'COHORT.json',dict(status='locked_before_P5_predictions',sources=128,queries=128,arrivals=768,conditions=CONDS,orders=1,availability=.25,expert_pairs=192,checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,selection='source then one query by prespecified SHA256; one independently hashed order',pool_sources=len(groups),pool_queries=len(meta),pool='local official HC-STVG-v2 validation confirm512',historical_project_exposure=True,GT_used_for_selection=False,annotation_intake='annotation-bearing JSON deserialized only to extract whitelisted query/media metadata; no GT-based selection or model access',parameter_support='same seed/rho=.05; center and resulting absolute radius from HC2 checkpoint, not Vid-trained center',resolution=224,nominal_sample_frames=64,time=time.time()))
    paths=[BASE/'PLAN.json',BASE/'EXPERT_PLAN.json',BASE/'SUBJECT_BARRIER.json',BASE/'COHORT.json']
    write(BASE/'LOCK.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in paths},checkpoint_sha256=CHECKPOINT_SHA,metadata_intake_path=str(annotation),metadata_intake_sha256=sha(annotation),time=time.time()))
    print('P5 locked:128 sources,768 arrivals,192 expert pairs; no inference',flush=True)
if __name__=='__main__':run()
