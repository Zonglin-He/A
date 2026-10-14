"""Read and verify explicit public manifests without traversing private assets."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--stage',required=True);s=p.add_subparsers(dest='action',required=True);c=s.add_parser('chunk');c.add_argument('--index',type=int,required=True);c.add_argument('--offset',type=int,required=True);v=s.add_parser('readback');v.add_argument('--commit',required=True);v.add_argument('--tree',required=True);v.add_argument('--receipt',required=True);a=p.parse_args();m=json.loads((R/a.manifest).read_text());rows=m if isinstance(m,list)else m['files'];stage=R/a.stage
 def data(row):
  rel=Path(row['path']);assert not rel.is_absolute()and'..'not in rel.parts;b=(stage/rel).read_bytes();assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'];return b
 if a.action=='chunk':
  row=rows[a.index];b=data(row);t=base64.b64encode(b).decode()if row.get('encoding')=='base64'else b.decode();print(json.dumps(dict(chunk=t[a.offset:a.offset+18000],total=len(t))));return
 gd=R.parent/'visual-grounding-public-A/.git'
 def git(*xs):return subprocess.check_output(['git','--git-dir='+str(gd),*xs])
 assert git('rev-parse',a.commit+'^{tree}').decode().strip()==a.tree;entries={}
 for line in git('ls-tree','-r','-z',a.commit).split(b'\0'):
  if line:
   meta,name=line.split(b'\t');mode,kind,h=meta.decode().split();entries[name.decode()]=(mode,kind,h)
 for row in rows:
  b=data(row);h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert entries[row['path']]==('100644','blob',h);assert git('cat-file','blob',h)==b
 out=dict(status='pass',scope='explicit public manifest exact bytes readback',remote_commit=a.commit,remote_tree=a.tree,files=len(rows),bytes=sum(x['bytes']for x in rows),all_remote_bytes_SHA256_Gitblob_mode_tree_exact=True,manifest_sha256=hashlib.sha256((R/a.manifest).read_bytes()).hexdigest());q=R/a.receipt
 with q.open('x')as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
