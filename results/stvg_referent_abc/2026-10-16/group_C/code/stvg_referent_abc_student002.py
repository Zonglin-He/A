"""Independent Teacher-policy replay and unchanged complete TA-STVG updates."""
from pathlib import Path
import ast,copy,gc,hashlib,inspect,json,sys,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import BASE,PRIOR,read,put,sha,verify
def teacher_root(arm):
    import numpy as np
    from scripts.stvg_referent_abc_tokens002 import select
    bar=read(BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==16 and bar['response_count']==64 and not bar['GT_read'];jobs={j['job_id']:j for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']};records=[]
    for f in bar['files']:
        assert sha(ROOT/f['path'])==f['sha256'];d=read(ROOT/f['path']);j=jobs[d['job_id']];cache=BASE/'native_cache'/j['job_id'];saved=read(cache/'POLICIES.json');assert d['job_binding_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()==saved['job_sha256'];assert d['native_policy_sha256']==sha(cache/'POLICIES.json')and not d['GT_read'];assert sha(cache/'native.npz')==saved['native_sha256']
        with np.load(cache/'native.npz',allow_pickle=False)as a:
            for i,(start,stop)in enumerate(((0,512),(512,1280))):
                v=a[f'valid{i}'];ma=a[f'mandatory{i}'];gr=a[f'groups{i}'];z=a['features'][start:stop].reshape(len(v),256,-1);fresh=select(z,a['query_embedding'],a['caption_embedding'],v,ma,gr);original=saved['records'][i]
                assert fresh['quotas']==original['quotas']and fresh['role_quotas']==original['role_quotas'];assert fresh['edges']==original['edges']
                for k in ('A','B','C'):assert fresh['indices'][k].tolist()==original['indices'][k]
                ix=np.array(original['indices'][arm]);assert v.reshape(-1)[ix].all()and set(np.flatnonzero(ma))<=set(ix)
                assert sum(original['quotas'])==len(ix)==int(__import__('math').ceil(.125*len(v)*256))
        assert d['sequence_readback']['all_nonvisual_tokens_preserved'];records.append(dict(case_id=j['case_id'],slot=j['center_slot'],selected_candidate_id=d['parsed'].get('selected_candidate_id'),parse_status=d['parsed']['status'],source_video_job_sha256=saved['job_sha256'],token_count_kept=sum(r['total_tokens_kept']for r in saved['records']),GT_read=False))
    put(BASE/'teacher'/arm/'ROOT_READBACK.json',dict(status='pass',count=16,response_count=64,records=records,all_three_native_policies_replayed_exactly=True,valid_area_ID_quotas_verified=True,GT_read=False,barrier_sha256=sha(BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json')))
def prepare(arm):
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.prepare_stvg_lite_reclocked_inputs002 import _anchor_from_teacher_selection
    verify();assert read(BASE/'teacher'/arm/'ROOT_READBACK.json')['status']=='pass';jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];refs=[];rr=read(BASE/'ROSTER_LOCK.json')['records']
    for rec in rr:
        cid=rec['probe_alias'];case,p=load_case(rec);pool=read(PRIOR/'spatial/E1/proposals'/f'{cid}.json');anchors=[];decisions=[]
        for j in [x for x in jobs if x['case_id']==cid]:
            p=BASE/'teacher'/arm/(j['job_id']+'.json');response=read(p);window=next(w for w in pool['windows']if w['slot']==j['center_slot']);a,reason=_anchor_from_teacher_selection(j,response,case['frame_ids'],window['source_grid_position'],teacher_output_sha256=sha(p));decisions.append(dict(slot=j['center_slot'],reason=reason,selected_candidate_id=response['parsed'].get('selected_candidate_id'),teacher_sha256=sha(p),GT_read=False))
            if a is not None:anchors.append(a)
        assert len({a['position']for a in anchors})==len(anchors);refs.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256'],anchors=anchors,decisions=decisions,GT_read=False));del case;gc.collect()
    b=BASE/'trials'/arm;b.mkdir(parents=True,exist_ok=False);put(b/'REFERENCE_LOCK.json',dict(status='locked',count=16,records=refs,GT_read=False,presence_threshold=.6,source_candidate_forbidden=True));(b/'TEMPORAL_LOCK.json').write_bytes((BASE/'TEMPORAL_LOCK.json').read_bytes())
    d=copy.deepcopy(read(PRIOR/'trials/E0/DESIGN_LOCK.json'));d.update(trial_id=arm,cohort='exposed_development16',scope='native evidence token policy '+arm+'; fixed Temporal and original M2 configuration; no optimizer changes',count=16);put(b/'DESIGN_LOCK.json',d)
    rt=copy.deepcopy(read(PRIOR/'trials/E0/RUNTIME.json'));rt.update(trial_id=arm,scope=d['scope'])
    for p in [Path(__file__),ROOT/'scripts/run_stvg_referent_abc002.py',BASE/'RUNTIME.json',BASE/'DESIGN_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'teacher'/arm/'ROOT_READBACK.json',BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json',b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json',b/'DESIGN_LOCK.json']:rt['pins'][str(p.relative_to(ROOT))]=sha(p)
    put(b/'RUNTIME.json',rt)
def check(arm):
    verify();rt=read(BASE/'trials'/arm/'RUNTIME.json')
    for rel,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/rel)==h,rel
def student(mode,arm):
    from scripts import run_stvg_instance_student32_v1 as old
    original=inspect.getsource(old.worker if mode=='worker'else old.root);tree=ast.parse(original)
    class Cohort(ast.NodeTransformer):
        def visit_Constant(self,node):return ast.copy_location(ast.Constant(16),node)if type(node.value)is int and node.value==32 else node
    tree=ast.fix_missing_locations(Cohort().visit(tree));s=ast.unparse(tree)
    if mode=='root':
        assert s.count("bb = arr(o['boxes'])")==1;s=s.replace("bb = arr(o['boxes'])","bb = arr(o['boxes']).astype(float)")
    ns=dict(old.__dict__);ns.update(BASE=BASE,check=check,modes=lambda _:('M2',));exec(compile(s,str(Path(__file__)),'exec'),ns);ns[mode](arm)
def run(mode,arm):
    if mode=='teacher_root':teacher_root(arm)
    elif mode=='student_prepare':prepare(arm)
    elif mode=='student_worker':student('worker',arm)
    elif mode=='student_root':student('root',arm)
    else:raise ValueError(mode)
