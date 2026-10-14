"""Transfer only root-approved public bytes; no private research payload traversal."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/stvg_support_aware_temporal_spatial_update_v1'
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True);c=sub.add_parser('chunk');c.add_argument('--index',type=int,required=True);c.add_argument('--offset',type=int,required=True);r=sub.add_parser('readback');r.add_argument('--commit',required=True);r.add_argument('--tree',required=True);r.add_argument('--receipt',required=True);a=p.parse_args();rows=json.loads((BASE/'PUBLIC_MANIFEST.json').read_text())
 if a.action=='chunk':
  v=rows[a.index];b=(BASE/'public_stage'/v['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==v['sha256']and len(b)==v['bytes'];s=base64.b64encode(b).decode()if v['encoding']=='base64'else b.decode();print(json.dumps(dict(chunk=s[a.offset:a.offset+18000],total=len(s))));return
 gitdir='/home/wwww/visual-grounding-public-A/.git'
 def git(*args):return subprocess.check_output(['git','--git-dir='+gitdir,*args])
 assert git('rev-parse',a.commit+'^{tree}').decode().strip()==a.tree;entries={}
 for line in git('ls-tree','-r','-z',a.commit).split(b'\0'):
  if line:
   meta,name=line.split(b'\t');mode,kind,digest=meta.decode().split();entries[name.decode()]=(mode,kind,digest)
 for v in rows:
  assert entries[v['path']]==(v['mode'],'blob',v['blob_sha']);remote=git('cat-file','blob',v['blob_sha']);local=(BASE/'public_stage'/v['path']).read_bytes();assert remote==local and len(remote)==v['bytes']and hashlib.sha256(remote).hexdigest()==v['sha256']
 out=dict(status='pass',scope='combined finite optimizer and global-grid P0 public files only',remote_commit=a.commit,remote_tree=a.tree,files=len(rows),bytes=sum(v['bytes']for v in rows),all_remote_bytes_SHA256_Gitblob_mode_tree_exact=True)
 (BASE/a.receipt).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
