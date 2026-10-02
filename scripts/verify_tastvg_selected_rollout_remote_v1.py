"""Read every published file back from the remote Git commit, then align checkout."""
import sys,json,time,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/tastvg_selected_rollout_v1'
PUBLIC=Path('/home/wwww/visual-grounding-public-A')
def git(*args):return subprocess.check_output(['git',*args],cwd=PUBLIC)
def run(commit):
 m=json.loads((BASE/'PUBLIC_MANIFEST.json').read_text());assert m['repository']=='Zonglin-He/A'
 assert git('remote','get-url','origin').decode().strip()=='https://github.com/Zonglin-He/A.git'
 git('fetch','origin','main');assert git('rev-parse','origin/main').decode().strip()==commit
 staged=git('diff','--cached','--name-only').decode().splitlines();allow={f['path'] for f in m['files']};assert set(staged)<=allow
 tree=git('write-tree').decode().strip();assert tree==git('rev-parse',commit+'^{tree}').decode().strip()
 checked=[]
 for f in m['files']:
  b=git('show',commit+':'+f['path']);assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'];checked.append({k:f[k] for k in ['path','bytes','sha256']})
 head=git('rev-parse','HEAD').decode().strip();assert head in [m['parent_sha'],commit]
 git('reset','--soft','origin/main');assert not git('status','--porcelain').strip()
 receipt=dict(status='pass',repository=m['repository'],commit=commit,tree_sha=tree,file_count=len(checked),total_bytes=sum(x['bytes'] for x in checked),files=checked,public_checkout_clean=True,time=time.time())
 (BASE/'REMOTE_READBACK.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['status','commit','file_count','total_bytes','public_checkout_clean']},indent=2))
if __name__=='__main__':run(sys.argv[1])
