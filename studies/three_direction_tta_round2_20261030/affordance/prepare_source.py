"""Rebuild the two needed modules from the official pinned checkout and reviewed patch."""
import argparse,pathlib,subprocess,hashlib,tempfile,shutil
ROOT=pathlib.Path(__file__).resolve().parent
COMMIT='8ab28ec4dd7bbfe4d3a70aa9fe9b7ef1295d0cef'
ORIGINAL={'GREAT.py':'c4e224aeb923b651e7355ee8c4fc41bae83e3122871ed61ff91c34b1cddbae15','pointnet2_utils.py':'65a0f468020c839dba5c2db30955b53d8dedf4215e707123e19b1b54ef17990c'}
EXPECTED={'GREAT.py':'71a337b266edc2ab1172f1c4ac1f2676678831965f308e2ca5cdea07dfda1339','pointnet2_utils.py':ORIGINAL['pointnet2_utils.py'],'__init__.py':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--upstream',type=pathlib.Path,default=ROOT.parent/'upstream');a.add_argument('--output',type=pathlib.Path,default=ROOT/'frozen_base');args=a.parse_args()
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.upstream,text=True).strip()==COMMIT
 with tempfile.TemporaryDirectory() as directory:
  tmp=pathlib.Path(directory);(tmp/'model').mkdir()
  for f,h in ORIGINAL.items():
   p=args.upstream/'model'/f;assert sha(p)==h,(f,'Original source changed');shutil.copyfile(p,tmp/'model'/f)
  subprocess.run(['patch','--batch','-p1','--directory',str(tmp),'-i',str(ROOT/'compat/GREAT_compat.patch')],check=True)
  (tmp/'model/__init__.py').touch();dest=args.output/'model';dest.mkdir(parents=True,exist_ok=True)
  for f,h in EXPECTED.items():
   p=tmp/'model'/f;assert sha(p)==h,(f,'Patch did not reconstruct executed source');target=dest/f
   if target.exists():assert target.read_bytes()==p.read_bytes(),('Refuse to overwrite different source',target)
   else:shutil.copyfile(p,target)
 print('Reconstructed exact executed baseline modules; no model or data loaded.')
if __name__=='__main__':main()
