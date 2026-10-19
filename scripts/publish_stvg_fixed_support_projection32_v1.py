"""Stage an explicit allowlist; verify connector dual-ref CAS and remote bytes.

Uses the existing Zonglin-He/A public checkout. Raw research artifacts, labels,
media and unrelated work are excluded. A retry reuses its immutable commit plan.
"""
from pathlib import Path
import ast,concurrent.futures,hashlib,json,os,re,subprocess,sys,tempfile,urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_fixed_support_projection32_v1 import D,P,verify
REPO=Path('/home/wwww/visual-grounding-public-A')
REFS=['refs/heads/main','refs/heads/research/stvg-opd-paper-hc2-revision-v2']


def git(*args,data=None,env=None):
    p=subprocess.run(['git','-C',str(REPO),*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,check=False)
    if p.returncode:raise RuntimeError(f'git {args[0]} failed: '+p.stderr.decode(errors='replace')[:1200])
    return p.stdout


def remote():
    return {b.decode():a.decode()for a,b in (line.split(b'\t')for line in git('ls-remote','--refs','origin',*REFS).splitlines())}


def unrelated():
    names=git('ls-files','--others','--exclude-standard','-z').split(b'\0')
    return {os.fsdecode(n):sha(REPO/os.fsdecode(n))for n in names if n and (REPO/os.fsdecode(n)).is_file()}


def main():
    verify();assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    pub=D/'publication';pub.mkdir(exist_ok=True)
    origin=git('remote','get-url','origin').decode().strip()
    assert origin in ['https://github.com/Zonglin-He/A.git','https://github.com/Zonglin-He/A','git@github.com:Zonglin-He/A.git']
    files=read(D/'PUBLIC_FILES_FINAL.json')['files']+['scripts/publish_stvg_fixed_support_projection32_v1.py','results/stvg_error_state/2026-10-19/fixed_support_projection32/ROUND_REVIEW.md','scripts/lock_stvg_fine_step_confirm32_v1.py','scripts/run_stvg_fine_step_confirm32_v1.py','protocols/stvg_fine_step_confirm32_v1.md','results/stvg_error_state/2026-10-19/fine_step_confirm32/PREPARED_CONFIRMATION.json'];assert len(files)==len(set(files))
    assert all(f.startswith(('vg_tta/','scripts/','protocols/','results/stvg_error_state/2026-10-19/fixed_support_projection32/','results/stvg_error_state/2026-10-19/fine_step_confirm32/'))for f in files)
    rows=read(D/'PLAN_LOCK.json')['records'];raw=[]
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']}
    for r in rows:
        raw.extend([r['parent'],plans[r['dataset']][r['query_ordinal']]['input']['caption']])
    sensitive=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
    manifest=[]
    for f in files:
        data=(ROOT/f).read_bytes();text=data.decode('utf8');assert not sensitive.search(text),f
        for value in raw:
            if len(value)>=8:assert json.dumps(value,ensure_ascii=False)not in text and repr(value)not in text,(f,'raw identifier/text')
        if f.endswith('.json'):json.loads(text,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
        if f.endswith('.py'):ast.parse(text,filename=f)
        manifest.append(dict(path=f,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
            git_blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    planfile=pub/'COMMIT_PLAN.json'
    if not planfile.exists():
        assert not git('diff','--name-only','HEAD').strip() and not git('diff','--cached','--name-only').strip()
        old=remote();assert len(old)==2 and len(set(old.values()))==1
        parent=old[REFS[0]];assert git('rev-parse','HEAD').decode().strip()==parent
        before=unrelated()
        if (pub/'UNRELATED_UNTRACKED_BEFORE.json').exists():assert before==read(pub/'UNRELATED_UNTRACKED_BEFORE.json')
        else:put(pub/'UNRELATED_UNTRACKED_BEFORE.json',before)
        # A private temporary INDEX protects the checkout index and unrelated files.
        index=pub/'private-export.index';assert not index.exists()
        env=dict(os.environ,GIT_INDEX_FILE=str(index));git('read-tree',parent,env=env)
        for rec in manifest:
            blob=git('hash-object','-w','--stdin',data=(ROOT/rec['path']).read_bytes()).decode().strip()
            assert blob==rec['git_blob'];git('update-index','--add','--cacheinfo','100644',blob,rec['path'],env=env)
        tree=git('write-tree',env=env).decode().strip()
        # Reuse this public repository's existing publication identity for this
        # command only; do not change the user's local/global Git configuration.
        identity=git('show','-s','--format=%an%n%ae%n%cn%n%ce',parent).decode().splitlines()
        assert len(identity)==4 and all(identity)
        authorenv=dict(os.environ,GIT_AUTHOR_NAME=identity[0],GIT_AUTHOR_EMAIL=identity[1],
                       GIT_COMMITTER_NAME=identity[2],GIT_COMMITTER_EMAIL=identity[3])
        commit=git('commit-tree',tree,'-p',parent,data=b'Diagnose frozen-support mean projection and prepare matched step confirmation\n',env=authorenv).decode().strip()
        assert set(git('diff-tree','--no-commit-id','--name-only','-r',commit).decode().splitlines())==set(files)
        plan=dict(commit=commit,tree=tree,parent=parent,old_refs=old,files=manifest)
        put(planfile,plan)
    else:
        plan=read(planfile);assert manifest==plan['files']
    if len(sys.argv)>1 and sys.argv[1]=='stage':
        print(json.dumps(dict(status='staged_no_remote_ref_mutation',commit=plan['commit'],tree=plan['tree'],files=len(manifest))),flush=True);return
    # Ref mutations use the existing authenticated GitHub connector with leases.
    # This verifier never attempts unauthenticated CLI HTTPS writes.
    assert remote()==dict.fromkeys(REFS,plan['commit'])
    git('fetch','origin',*REFS)
    assert git('rev-parse',plan['commit']+'^{tree}').decode().strip()==plan['tree']
    def check(rec):
        url='https://raw.githubusercontent.com/Zonglin-He/A/'+plan['commit']+'/'+rec['path']
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'STVG-publication-audit'}),timeout=60)as r:
            assert r.status==200;data=r.read()
        assert len(data)==rec['bytes']and hashlib.sha256(data).hexdigest()==rec['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==rec['git_blob']
        listing=git('ls-tree',plan['commit'],'--',rec['path']).decode().strip();mode,kind,blobpath=listing.split(' ',2)
        blob,path=blobpath.split('\t');assert mode=='100644'and kind=='blob'and blob==rec['git_blob']and path==rec['path']
        assert git('show',plan['commit']+':'+rec['path'])==data
        return dict(**rec,HTTP_bytes_SHA256_Gitblob='pass',fetched_mode_tree='pass')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:verified=list(pool.map(check,manifest))
    if git('rev-parse','HEAD').decode().strip()!=plan['commit']:
        assert not git('diff','--name-only','HEAD').strip()
        git('merge','--ff-only',plan['commit'])
    assert not git('diff','--name-only','HEAD').strip()and not git('diff','--cached','--name-only').strip()
    assert unrelated()==read(pub/'UNRELATED_UNTRACKED_BEFORE.json')
    verify()
    receipt=dict(status='pass',count=32,commit=plan['commit'],tree=plan['tree'],parent=plan['parent'],
        files_count=len(verified),bytes=sum(f['bytes']for f in verified),files=verified,
        both_refs_fresh_CAS=True,all_actual_HTTP_bytes_SHA256_Gitblob_fetched_mode_tree=True,
        unrelated_untracked_preserved=True,tracked_and_index_clean=True,registries_unchanged=True,
        private_media_GT_coordinates_weights_raw_candidates_and_original_text_excluded=True,
        frozen_support_projection_diagnostic=True,new_source_confirmation_executed=False,original_predictions_unchanged=True,
        independent_confirmation=False,new_Source_captures_completed=read(P/'COST_BREAKDOWN.json')['total_Source_captures_completed'],
        new_DINO_encoding_frames_completed=sum(read(P/'COST_BREAKDOWN.json')[k]for k in ['final_worker_DINO_encodings_completed','engineering_DINO_encodings_completed']),
        new_original_DINO_detection_calls=0,no_GT_selected_deployment_steps=True)
    put(pub/'FINAL_GITHUB_RECEIPT.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items()if k!='files'}),flush=True)


if __name__=='__main__':main()
