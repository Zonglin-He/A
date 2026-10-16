"""Exact original episodic32 worker, changing only frozen WHO references."""
from pathlib import Path
import copy,gc,inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_global32_context_v1 import BASE,CORE,read,put,sha,verify
def prepare():
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.prepare_stvg_lite_reclocked_inputs002 import _anchor_from_teacher_selection
    verify();assert read(BASE/'teacher/G32/ROOT_READBACK.json')['status']=='pass';jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];rr=read(BASE/'ROSTER_LOCK.json')['records'];source=[];refs=[]
    original_sources={r['case_id']:r for r in read(CORE/'SOURCE_CAPTURE_BARRIER.json')['files']}
    for rec in rr:
        cid=rec['probe_alias'];c,p=load_case(rec);f=original_sources[cid];assert sha(p)==f['sha256'];source.append(f);anchors=[];decisions=[];jj=[j for j in jobs if j['case_id']==cid];pool=read(CORE/'spatial/E1/proposals'/(cid+'.json')) if jj else None
        for j in jj:
            rp=BASE/'teacher/G32'/(j['job_id']+'.json');d=read(rp);w=next(x for x in pool['windows'] if x['slot']==j['center_slot']);a,why=_anchor_from_teacher_selection(j,d,c['frame_ids'],w['source_grid_position'],teacher_output_sha256=sha(rp))
            if a is not None:anchors.append(a)
            decisions.append(dict(slot=j['center_slot'],reason=why,selected_candidate_id=d['parsed'].get('selected_candidate_id'),teacher_sha256=sha(rp),GT_read=False))
        refs.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=c['source_case_sha256'],anchors=anchors,decisions=decisions,GT_read=False));del c;gc.collect()
    put(BASE/'SOURCE_CAPTURE_BARRIER.json',dict(status='complete',count=32,files=source,exact_core_reuse=True,GT_read=False))
    b=BASE/'trials/G32';b.mkdir(parents=True,exist_ok=False);put(b/'REFERENCE_LOCK.json',dict(status='locked',count=32,records=refs,source_candidate_forbidden=True,presence_threshold=.6,GT_read=False))
    ids={r['probe_alias'] for r in rr};tt=copy.deepcopy(read(CORE/'TEMPORAL_LOCK.json'));tt.update(count=32,records=[r for r in tt['records'] if r['case_id'] in ids]);assert len(tt['records'])==32;put(b/'TEMPORAL_LOCK.json',tt)
    d=copy.deepcopy(read(CORE/'trials/A/DESIGN_LOCK.json'));d.update(trial_id='G32',count=32,cohort='fixed_hash16_parent_per_domain',scope='only full-input global32_keep512; exact core A local evidence/OPD',family_reuse_from=None);put(b/'DESIGN_LOCK.json',d)
    rt=copy.deepcopy(read(CORE/'trials/A/RUNTIME.json'));rt.update(trial_id='G32',scope=d['scope'])
    for p in [Path(__file__),BASE/'EXECUTION_RUNTIME.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'ROSTER_LOCK.json',BASE/'DESIGN_LOCK.json',BASE/'teacher/G32/ROOT_READBACK.json',b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json',b/'DESIGN_LOCK.json']:rt['pins'][str(p.relative_to(ROOT))]=sha(p)
    put(b/'RUNTIME.json',rt)
def check(tid):
    assert tid=='G32';verify();rt=read(BASE/'trials/G32/RUNTIME.json')
    for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
def execute(mode):
    from scripts import run_stvg_instance_student32_v1 as old
    src=inspect.getsource(old.worker if mode=='worker' else old.root)
    if mode=='root':assert src.count("bb=arr(o['boxes'])")==1;src=src.replace("bb=arr(o['boxes'])","bb=arr(o['boxes']).astype(float)")
    ns=dict(old.__dict__);ns.update(BASE=BASE,check=check,modes=lambda _:('M2',));exec(compile(src,str(Path(__file__)),'exec'),ns);ns[mode]('G32')
if __name__=='__main__':
    mode=sys.argv[1];prepare() if mode=='prepare' else execute(mode)
