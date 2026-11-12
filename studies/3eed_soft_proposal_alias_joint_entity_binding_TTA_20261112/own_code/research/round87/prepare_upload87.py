"""Prepare safe public blob identities after the independent export audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'research/round76'))
from common76 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,default=1);a=ap.parse_args();assert a.attempt>=1;tag=str(a.attempt).zfill(3)
H=ROOT/'research/round87';pub=H/('publication/public_export_attempt'+tag);audit=read(H/('independent_public_audit_attempt'+tag+'/AUDIT_RESULT.json'));assert audit['state'].startswith('PASS_') and audit['privacy_findings']==0
manifest=read(pub/'MANIFEST.json');files=[]
for f in sorted(pub.rglob('*')):
 if not f.is_file():continue
 assert not f.is_symlink();b=f.read_bytes();rel=f.relative_to(pub).as_posix();files.append({'path':rel,'local_path':f.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()})
assert len(files)==audit['files'] and sum(f['bytes'] for f in files)==audit['bytes'];out=H/'publication/UPLOAD_INDEX_PRIVATE.json';assert not out.exists();dump(out,{'prefix':manifest['prefix'],'files':files,'publication_base_commit':read(H/'publication/BASE_SNAPSHOT_PRIVATE.json')['commit'],'independent_public_audit_sha256':sha(H/('independent_public_audit_attempt'+tag+'/AUDIT_RESULT.json'))});print({'files':len(files),'bytes':sum(f['bytes'] for f in files)})
