"""Compare downloaded Git objects with local byte manifest; no inference/scoring."""
import subprocess,json,hashlib,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_hierarchical_evidence_guided_v1'
p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--tree',required=True);p.add_argument('--git-dir',default=str(BASE/'remote_readback.git'));p.add_argument('--receipt',default='FINAL_GITHUB_RECEIPT.json');a=p.parse_args()
gitdir=Path(a.git_dir)
def git(*args):return subprocess.check_output(['git','--git-dir='+str(gitdir),*args])
assert git('rev-parse',a.commit+'^{tree}').decode().strip()==a.tree
entries={}
for z in git('ls-tree','-r','-z',a.commit).split(b'\0'):
 if not z:continue
 left,path=z.split(b'\t',1);mode,typ,h=left.decode().split();entries[path.decode()]={'mode':mode,'type':typ,'blob_sha':h}
m=json.loads((BASE/'PUBLIC_MANIFEST.json').read_text());rows=[]
for v in m:
 e=entries[v['path']];assert all(e[k]==v[k] for k in ['mode','type','blob_sha'])
 data=git('cat-file','blob',e['blob_sha']);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
 assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==e['blob_sha']
 rows.append({**v,'remote_bytes_SHA256_Gitblob_mode_tree':'pass'})
out={'status':'pass','scope':'actual_downloaded_remote_Git_objects_per_file_bytes_SHA256_Gitblob_mode_tree','commit':a.commit,'tree':a.tree,'count':len(m),'bytes':sum(v['bytes'] for v in m),'records':rows,'CPU_only':True,'GT_read':False,'time':time.time()}
(BASE/a.receipt).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'}))
