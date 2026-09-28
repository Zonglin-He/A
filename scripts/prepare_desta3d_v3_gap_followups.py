"""CPU-only conditional package and metadata lock. Never reads labels_json."""
import hashlib,json,re,sqlite3,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_oracle_mixer_gap import D,ROSTER,OLD
from vg_tta.desta3d_v3_oracle_io import read,write,sha,local_dependencies
P=D.parent/'gap_followups_v1'
PROTO=ROOT/'protocols/desta3d_v3_gap_followups_v1.md'
KEYS={'source','source_id','parent','parent_id','video_id','original_video_id','video_sha256','media_sha256','key','query_key','query_id','sources','parents','source_ids','parent_ids','keys','selected_parents','excluded_parents'}

def identity_values(obj):
    """Whitelist identifiers only, never inspect loss/metric/annotation values."""
    values=set()
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in KEYS:
                    if isinstance(v,(str,int)):values.add(str(v))
                    elif isinstance(v,list):values.update(str(z) for z in v if isinstance(z,(str,int)))
                if k not in {'metrics','scores','labels','labels_json','annotations','response','boxes_xyxy','box_valid','event_active','GT','gt'} and isinstance(v,(dict,list)):walk(v)
        elif isinstance(x,list):
            for y in x:
                if isinstance(y,(dict,list)):walk(y)
    walk(obj);return values

def main():
    assert not P.exists()
    db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True)
    def authorize(action,tbl,col,dbn,trigger):
        if action==sqlite3.SQLITE_READ and col in ('labels_json','provenance_json'):return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    db.set_authorizer(authorize)
    source=[]
    for key,parent,split,domain,row_json in db.execute('select key,parent,split,domain,row_json from examples'):
        r=json.loads(row_json);source.append(dict(key=key,parent=parent,split=split,domain=domain,
            video_sha256=r['input']['video_sha256'],original_video_id=str(r['input']['original_video_id']),frames=len(r['input']['frame_ids'])))
    db.close()
    blocked=set();train=[r for r in source if r['split']=='train']
    for r in train:blocked.update(str(r[k]) for k in ('key','parent','video_sha256','original_video_id'))
    registry=[];skipped=[]
    names=re.compile(r'^(?:.*_)?(?:INPUTS|MANIFEST|COHORT|ROSTER|PANEL|SELECTION)\.json$',re.I)
    master=ROOT/'artifacts/desta3d_v3/full_source_roster_v1/INPUTS.json'
    for path in sorted((ROOT/'artifacts').rglob('*.json')):
        if not names.match(path.name):continue
        if any(x in str(path.relative_to(ROOT)).lower() for x in ('github_public','gap_followups','source_intake','source_media_audit','hc_media_repair')):continue
        if path==master:skipped.append(dict(path=str(path),reason='complete population inventory is not experimental exposure'));continue
        if any(x in path.name.upper() for x in ('CODE','RESULT','SCORE','ARTIFACT','DOWNLOAD','PRE_EXECUTION','FIGURE','SCAN_MANIFEST')):
            skipped.append(dict(path=str(path),reason='non-cohort inventory/code/result manifest'));continue
        # Whole-source acquisition manifests enumerate unrun validation data;
        # their train partition is conservatively excluded from the SQL above.
        if '/desta3d_v3/source_data_' in str(path) or '/desta3d_v3/source_full_' in str(path):
            skipped.append(dict(path=str(path),reason='full source acquisition inventory'));continue
        if path.stat().st_size>128*2**20:raise RuntimeError('Large metadata requires explicit schema audit: '+str(path))
        identifiers=identity_values(read(path));blocked.update(identifiers)
        registry.append(dict(path=str(path),sha256=sha(path),identity_count=len(identifiers)))
    # Prior provenance-only source-ID scan includes old development files which
    # did not use the later INPUTS naming convention. It contains no task scores.
    prior_audit=ROOT/'artifacts/c1_fresh_confirmation_v1/vid_train_audit/AUDIT.json'
    if prior_audit.exists():
        old_provenance=read(prior_audit)
        qualification=prior_audit.with_name('QUALIFICATION.json')
        qualified=read(qualification)
        inventory=ROOT/'artifacts/marginal_confirmation_v1_intake/source_inventory.json'
        assert qualified['audit_sha']==sha(prior_audit) and qualified['inventory_sha']==sha(inventory)
        # Existing audited exception permits only the official all-source ID
        # lists, never experimental mentions in the rest of that inventory.
        assert qualified['metadata_only_exception']==[
            'vidstg_vidor.official.vidstg.annotations.train_annotations.source_ids',
            'vidstg_vidor.official.vidstg.files.train_files.source_ids']
        identities=(set(old_provenance['matches'])|set(old_provenance['media']))-set(qualified['eligible_sources'])
        reserved_path=prior_audit.with_name('PROSPECTIVE_SOURCE_RESERVATION.json')
        reserved=read(reserved_path);identities.update(reserved['warmup']);identities.update(reserved['confirmation'])
        blocked.update(identities)
        registry.append(dict(path=str(qualification),sha256=sha(qualification),purpose='prior audited metadata-only official-inventory exception'))
        registry.append(dict(path=str(reserved_path),sha256=sha(reserved_path),purpose='conservatively exclude even old unrun reservation'))
        registry.append(dict(path=str(prior_audit),sha256=sha(prior_audit),identity_count=len(identities),purpose='conservative prior mentions/media, not proof all were experiments'))
    # Exact required current/prior source manifests are always explicit entries.
    required=[OLD/'TRAIN_INPUTS.json',OLD/'VALIDATION_INPUTS.json',D/'INPUTS.json',
        ROOT/'artifacts/desta3d_v1/SOURCE_INPUTS.json',ROOT/'artifacts/desta3d_v1/tta_run/target64_v1/INPUTS.json',
        ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2/INPUTS.json']
    assert all(str(p) in {r['path'] for r in registry} for p in required)
    candidates=[r for r in source if r['split']=='validation' and r['domain']=='Vid']
    by={p:[r for r in candidates if r['parent']==p] for p in sorted({r['parent'] for r in candidates})}
    excluded={p for p,rr in by.items() if any(str(r[k]) in blocked for r in rr for k in ('key','parent','video_sha256','original_video_id'))}
    eligible=sorted(set(by)-excluded,key=lambda p:hashlib.sha256(('gap-fresh-confirmation-v1|'+p).encode()).hexdigest())
    assert len(eligible)>=31,dict(eligible=len(eligible),excluded=len(excluded))
    chosen=set(eligible[:31]);fresh=[r for r in candidates if r['parent'] in chosen];fresh.sort(key=lambda r:(r['parent'],r['key']))
    oldrows=read(D/'INPUTS.json');oldparents={r['source'] for r in oldrows}
    assert not chosen&oldparents
    diagnosis=sorted(oldparents,key=lambda x:hashlib.sha256(('gap-rescue-source16-v1|'+x).encode()).hexdigest())[:16]
    rescue=[]
    for p in diagnosis:
        i,r=min(((i,r) for i,r in enumerate(oldrows) if r['source']==p),key=lambda z:hashlib.sha256(('gap-rescue-source16-v1|'+z[1]['key']).encode()).hexdigest())
        rescue.append(dict(index=i,key=r['key'],source=p,video_sha256=r['input']['video_sha256']))
    # Controlled environment preflight runs only CPU synthetic contracts.
    test=subprocess.run([sys.executable,'-B','-m','pytest','-q','tests/test_desta3d_v3_gap_candidates.py','tests/test_desta3d_v3_gap_decision.py'],cwd=ROOT,text=True,capture_output=True)
    assert test.returncode==0,(test.stdout,test.stderr)
    write(P/'EXCLUSION_REGISTRY.json',dict(manifests=registry,skipped=skipped,sql_train_parents=len({r['parent'] for r in train}),
        identity_values=len(blocked),remaining_validation_parents=len(eligible),excluded_validation_parents=len(excluded),
        scope='conservative enumerated metadata exclusion, not unverifiable pretraining/global-history nonmembership',labels_read=False,metrics_inspected=False))
    write(P/'FRESH_CONFIRMATION_ROSTER.json',dict(status='metadata_only_locked_unexecuted',parents=31,queries=len(fresh),rows=fresh,
        salt='gap-fresh-confirmation-v1',exclusion_registry_sha=sha(P/'EXCLUSION_REGISTRY.json'),labels_read=False,model_run=False))
    write(P/'RESCUE16_ROSTER.json',dict(status='metadata_only_locked_unexecuted',rows=rescue,parents=16,queries=16,salt='gap-rescue-source16-v1',outcome_selected=False))
    write(P/'CPU_PREFLIGHT.json',dict(status='passed',output=test.stdout,GPU=False,real_PTD_interface=False))
    paths=local_dependencies([Path(__file__),PROTO,ROOT/'vg_tta/desta3d_v3_gap_candidates.py',ROOT/'scripts/report_desta3d_v3_gap_decision.py',
        ROOT/'tests/test_desta3d_v3_gap_candidates.py',ROOT/'tests/test_desta3d_v3_gap_decision.py',
        P/'EXCLUSION_REGISTRY.json',P/'FRESH_CONFIRMATION_ROSTER.json',P/'RESCUE16_ROSTER.json',P/'CPU_PREFLIGHT.json',D/'LOCK.json'])
    write(P/'LOCK.json',dict(pins={str(p):sha(p) for p in paths}));write(P/'REGISTRATION.json',dict(time=time.time(),GPU=False,protocol_sha=sha(PROTO),status='conditional_CPU_package_no_new_scientific_launch',audit_live_lock_sha=sha(D/'LOCK.json')))
    print(json.dumps(dict(fresh_parents=31,fresh_queries=len(fresh),eligible_parents=len(eligible),rescue_queries=16,CPU=test.stdout)))
if __name__=='__main__':main()
