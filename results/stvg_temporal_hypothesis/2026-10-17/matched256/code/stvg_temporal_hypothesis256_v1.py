"""Same B, original core256; exact prior32 reuse, remaining224 only."""
from pathlib import Path
import copy,gc,inspect,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_temporal_hypothesis256_v1'
OLD=ROOT/'artifacts/stvg_temporal_hypothesis32_v1'
T2=ROOT/'artifacts/stvg_temporal_t2_vs_t4_v1'
CORE=ROOT/'artifacts/stvg_spatial_controls256_v1'
def verify():
    for p,h in read(BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    if (BASE/'ACTIVE_EXECUTION.json').exists():assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
def init():
    from scripts.stvg_temporal_hypothesis_fit32_v1 import CFG
    assert not BASE.exists();BASE.mkdir()
    for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json']:(BASE/n).write_bytes((T2/n).read_bytes())
    rr=read(BASE/'ROSTER_LOCK.json')['records'];old={r['probe_alias']:r for r in read(OLD/'ROSTER_LOCK.json')['records']};assert len(rr)==256 and len(old)==32
    assert {ds:sum(r['dataset']==ds for r in rr) for ds in ['hc2','vidstg']}=={'hc2':128,'vidstg':128}
    for r in rr:
        if r['probe_alias'] in old:
            o=old[r['probe_alias']]
            for k in ['parent_sha256','query_key_sha256','source_case_path','source_checkpoint_sha256']:assert r[k]==o[k],k
    for n in ['PRIMARY_ROOT_READBACK.json','PRIMARY_COMPLETION.json','TEACHER_ROOT_READBACK.json']:assert read(OLD/n)['count']==32
    for p,h in read(OLD/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    src=read(BASE/'SOURCE_CAPTURE_BARRIER.json');assert src['count']==256
    for f in src['files']:assert sha(ROOT/f['path'])==f['sha256']
    bp=read(T2/'GLOBAL_PREDICTION_BARRIER.json');assert bp['status']=='complete' and bp['count']==256
    for f in bp['files']:assert sha(ROOT/f['path'])==f['sha256']
    write(BASE/'REUSE_LOCK.json',dict(status='locked',count=32,case_ids=sorted(old),rule='complete source/hypotheses/input/Teacher/fitter/readout exact; no score routing',old_result_sha256=sha(OLD/'POSTSEAL_RESULTS.json'),GT_read=False))
    write(BASE/'METHOD_LOCK.json',dict(status='locked',count=256,per_domain=128,old32_reused=32,new_sources=224,config=CFG,scientific_fitter_sha256=sha(ROOT/'scripts/stvg_temporal_hypothesis_fit32_v1.py'),unchanged_B=True,Temporal_only=True,Spatial_fits=0,external_Actionness=False,parameters=512,max_new_description_calls=448,max_new_consistency_calls=448,extra_prefix_qualifications=2,CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),GT_read=False))
    paths=[Path(__file__),ROOT/'scripts/continue_stvg_temporal_hypothesis256_v1.py',ROOT/'scripts/score_stvg_temporal_hypothesis256_v1.py',ROOT/'protocols/stvg_temporal_hypothesis256_v1.md',ROOT/'methods/CURRENT_METHOD.json']
    paths += [ROOT/'scripts'/n for n in ['stvg_temporal_hypothesis32_v1.py','stvg_temporal_hypothesis_fit32_v1.py','stvg_temporal_semantic_view_fit32_v1.py','stvg_evidence_temporal_fit_v1.py','stvg_temporal_semantic_grounder32_v1.py','stvg_temporal_semantic_grounder32_v2.py','stvg_table1_temporal_adapter_v1.py','run_stvg_referent_abc002.py','stvg_qwen_uniform_clock002.py','run_decota_paper_main_v1.py','stvg_pipeline_speed_v1.py','stvg_factorial_native256_v1.py','launch_stvg_instance_identity_v1.py']]
    paths += [BASE/n for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','METHOD_LOCK.json','REUSE_LOCK.json']]
    paths += [T2/n for n in ['GLOBAL_PREDICTION_BARRIER.json','ROOT_READBACK.json','POSTSEAL_RESULTS.json']]
    paths += [OLD/n for n in ['INPUT_BARRIER.json','TEACHER_BARRIER.json','FIT_BARRIER.json','TARGET_BARRIER.json','PRIMARY_GLOBAL_PREDICTION_BARRIER.json']]
    write(BASE/'EXECUTION_RUNTIME.json',dict(status='locked',count=256,pins={str(p.relative_to(ROOT)):sha(p) for p in paths},GT_read=False));write(BASE/'STATUS.json',dict(status='prepared',count=256,GT_read=False),mutable=True)
def prepare():
    import numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case,_render_marked_frames
    from scripts import stvg_temporal_hypothesis_fit32_v1 as f,stvg_factorial_native256_v1 as rgb
    verify();sys.addaudithook(guard);rgb.BASE=BASE;oldjobs={j['case_id']:j for j in read(OLD/'INPUT_BARRIER.json')['jobs']};rows=[];files=[];proofs=[]
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        c,_=load_case(r);cid=r['probe_alias'];req=read(CORE/'temporal/requests'/(cid+'.json'));available=sorted(x['physical_frame_id'] for x in req['frame_map']);off=f.native.offsets(c,_hidden_from_source_case(c));selection=f.select(off,available);assert selection['Native_interval']==list(c['Native_interval']);hs=selection['hypotheses']
        if cid in oldjobs:
            old=oldjobs[cid];assert sha(ROOT/old['path'])==old['sha256'];j=read(ROOT/old['path']);assert j['query_text']==req['query_text'] and j['source_case_sha256']==c['source_case_sha256']
            assert len(hs)==len(j['selection']['hypotheses'])
            for a,b in zip(hs,j['selection']['hypotheses']):
                for k in ['pairs','interval_frames','sampled_physical_frame_ids','native_joint_score']:assert a[k]==b[k],(cid,k)
                assert sha(ROOT/b['video_path'])==b['video_sha256']
            rows.append(dict(old,reused_prior32=True));proofs.append(dict(case_id=cid,status='exact_old_input_selection_and_source',GT_read=False))
        else:
            if len(hs)==2:
                rgb.CURRENT_CASE=c;pixels=rgb.native_rgb(req,set(x for h in hs for x in h['sampled_physical_frame_ids']))
                for k,h in enumerate(hs):
                    h['canonical_id']='Native' if k==0 else 'Alternative';frames,meta=_render_marked_frames(c,h,h['sampled_physical_frame_ids'],pixels);p=BASE/'inputs'/cid/(str(k)+'.npy');p.parent.mkdir(parents=True,exist_ok=True);np.save(p,frames,allow_pickle=False);h.update(video_path=str(p.relative_to(ROOT)),video_sha256=sha(p),metadata=meta);files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
                if hs[0]['video_sha256']==hs[1]['video_sha256'] and hs[0]['metadata']['clip_fps']==hs[1]['metadata']['clip_fps']:selection['status']='identical_actual_rendered_observations_no_retry'
            j=dict(case_id=cid,dataset=r['dataset'],query_text=req['query_text'],source_fps=float(c['input']['fps']),legal_bounds=[available[0],available[-1]+1],selection=selection,source_case_sha256=c['source_case_sha256'],GT_read=False);p=BASE/'inputs'/cid/'JOB.json';write(p,j);rows.append(dict(case_id=cid,path=str(p.relative_to(ROOT)),sha256=sha(p),reused_prior32=False))
        write(BASE/'INPUT_STATUS.json',dict(status='running',done=len(rows),total=256,count=256,GT_read=False),mutable=True);print('HYP256_INPUT',len(rows),flush=True);del c;gc.collect()
    write(BASE/'INPUT_BARRIER.json',dict(status='complete',count=256,jobs=rows,files=files,GT_read=False));write(BASE/'PRIOR32_INPUT_READBACK.json',dict(status='pass',count=32,records=proofs,GT_read=False))
def teacher():
    from scripts import stvg_temporal_hypothesis32_v1 as old
    reused={f['case_id']:f for f in read(OLD/'TEACHER_BARRIER.json')['files']}
    def reuse(cid,jf):
        f=reused[cid];assert sha(ROOT/f['path'])==f['sha256'];d=read(ROOT/f['path']);assert d['input_sha256']==jf['sha256'];return dict(f,reused_prior32=True)
    src=inspect.getsource(old.teacher);needle="cid=j['case_id'];hs=j['selection']['hypotheses'];ev=[]";assert src.count(needle)==1
    src=src.replace(needle,needle+"\n        if jf.get('reused_prior32'):\n            outputs.append(reuse(cid,jf));continue")
    src=src.replace('total=32,count=32','total=256,count=256').replace("count=32,files=outputs","count=256,files=outputs")
    ns=dict(old.__dict__);ns.update(BASE=BASE,verify=verify,reuse=reuse);exec(compile(src,__file__,'exec'),ns);ns['teacher']()
def teacher_root():
    from scripts import stvg_temporal_hypothesis32_v1 as old
    src=inspect.getsource(old.teacher_root).replace('count=32','count=256');ns=dict(old.__dict__);ns.update(BASE=BASE,verify=verify);exec(compile(src,__file__,'exec'),ns);ns['teacher_root']()
def fit():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_hypothesis_fit32_v1 as f
    verify();sys.addaudithook(guard);assert read(BASE/'SIGNAL_COMPLETION.json')['count']==256;jm={r['case_id']:r for r in read(BASE/'INPUT_BARRIER.json')['jobs']};tm={r['case_id']:r for r in read(BASE/'TEACHER_BARRIER.json')['files']};of={r['case_id']:r for r in read(OLD/'FIT_BARRIER.json')['files']};ot={r['case_id']:r for r in read(OLD/'TARGET_BARRIER.json')['files']};files=[];targets=[];checks=0
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=r['probe_alias']
        if jm[cid]['reused_prior32']:
            for v in [of[cid],ot[cid]]:assert sha(ROOT/v['path'])==v['sha256']
            files.append(dict(of[cid],reused_prior32=True));targets.append(dict(ot[cid],reused_prior32=True));continue
        c,_=load_case(r);j=read(ROOT/jm[cid]['path']);assert sha(ROOT/tm[cid]['path'])==tm[cid]['sha256'];d=read(ROOT/tm[cid]['path']);off=f.native.offsets(c,_hidden_from_source_case(c));hs=j['selection']['hypotheses'];rew=[e['reward'] for e in d['evidence']];ff=f.fit(off,hs,rew);checks+=f.audit(off,ff);v=list(c['Native_interval']) if ff['no_information'] else hs[int(np.argmax(ff['Teacher_conditional']))]['interval_frames'];tp=save_pt(BASE/'targets'/(cid+'.pt'),dict(view_intervals=dict(HYP=v),GT_read=False));tp['case_id']=cid;targets.append(tp);p=save_pt(BASE/'fits'/(cid+'.pt'),dict(case_id=cid,dataset=r['dataset'],fits=dict(HYP=ff),source_case_sha256=c['source_case_sha256'],GT_read=False));p['case_id']=cid;files.append(p);write(BASE/'FIT_STATUS.json',dict(status='running',done=len(files),total=256,count=256,GT_read=False),mutable=True);del c,ff;gc.collect()
    write(BASE/'TARGET_BARRIER.json',dict(status='complete',count=256,files=targets,GT_read=False));write(BASE/'FIT_BARRIER.json',dict(status='complete',count=256,files=files,independent_checks=checks,reused_prior32=32,GT_read=False))
def readout():
    from scripts import stvg_temporal_semantic_grounder32_v1 as old
    from scripts.stvg_temporal_semantic_grounder32_v2 import clean
    verify();clean();rr=read(BASE/'ROSTER_LOCK.json');oldids=set(read(BASE/'REUSE_LOCK.json')['case_ids']);op={r['case_id']:r for r in read(OLD/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']}
    def read_new(p):
        if Path(p)==BASE/'ROSTER_LOCK.json':return dict(rr,records=[r for r in rr['records'] if r['probe_alias'] not in oldids])
        return read(p)
    def write_full(p,d,**kw):
        if Path(p).name=='PRIMARY_GLOBAL_PREDICTION_BARRIER.json':
            for f in op.values():assert sha(ROOT/f['path'])==f['sha256']
            d['files']+= [dict(op[r['probe_alias']],reused_prior32=True) for r in rr['records'] if r['probe_alias'] in oldids];assert len(d['files'])==256
        if 'count' in d:d['count']=256
        if 'total' in d:d['total']=224
        return write(p,d,**kw)
    src=inspect.getsource(old.readout).replace("['SEM','DIRECT','PARA']","['HYP']")
    ns=dict(old.__dict__);ns.update(BASE=BASE,verify=verify,read=read_new,write=write_full);exec(compile(src,__file__,'exec'),ns);ns['readout']();write(BASE/'PRIMARY_COMPLETION.json',dict(status='complete',count=256,old32_reused=True,GT_read=False))
if __name__=='__main__':globals()[sys.argv[1]]()
