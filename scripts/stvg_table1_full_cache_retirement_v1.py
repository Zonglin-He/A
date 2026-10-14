"""Lossless custody and retirement of this FULL run's reproducible bridge media.

Only exact files named by a completed tracking batch are eligible, after its
teacher, Student and independent root seals. All 64-case historical caches are
excluded. Model-generated mask JSON is archived and byte-read back, never lost.
"""
from pathlib import Path
import argparse, hashlib, json, os, tarfile, time
ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / 'artifacts/stvg_final_table1_v1/FULL'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
    if p.exists():
        assert p.read_bytes()==data, p
        return
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('xb') as f: f.write(data);f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink()
def rel(p): return str(Path(p).resolve().relative_to(ROOT))

def retire(tracking_path, teacher_path, student_path, root_path):
    from scripts import run_stvg_table1_full_tracking_v1 as tracker
    tp=Path(tracking_path).resolve(); b=read(tp)
    assert b['schema']==tracker.FULL_BATCH_SCHEMA and b['status']=='complete' and b['GT_read'] is False
    selection=b['selection_sha256']; out=FULL/'CACHE_RETIREMENTS'/(selection+'.json')
    locked={'tracking_runtime_sha256':sha(FULL/'TRACKING_RUNTIME.json')}
    if out.exists():
        tracker._validate_retirement(tp,b,locked)
        return {'status':'already_verified','selection_sha256':selection,'GT_read':False}
    seals={}
    for kind,p in [('teacher_batch',teacher_path),('student_batch',student_path),('root_batch',root_path)]:
        p=Path(p).resolve();v=read(p)
        assert p.is_relative_to(FULL) and v['status'] in ('complete','sealed','pass') and v['GT_read'] is False
        assert v['count']==b['count'],kind
        if kind!='root_batch': assert v['aliases']==b['aliases'],kind
        else: assert v.get('student_barrier_sha256',v.get('barrier_sha256'))==sha(student_path)
        seals[kind]={'path':rel(p),'sha256':sha(p)}
    protected_files,protected_dirs=tracker._preflight_protected_cache_roots()
    def eligible(relative):
        p=(ROOT/relative).resolve()
        assert p.is_relative_to(tracker.CACHE_ROOT.resolve()) and not (ROOT/relative).is_symlink()
        return p not in protected_files and not any(p==d or p.is_relative_to(d) for d in protected_dirs)
    journal=FULL/'CACHE_RETIREMENT_PREPARED'/(selection+'.json')
    if journal.exists():
        plan=read(journal);assert plan['downstream_seals']==seals and plan['tracking_batch_barrier_sha256']==sha(tp)
        tracker._read_archive_members(plan['lossless_evidence_archive'])
    else:
        media={}; evidence={}
        for row in b['records']:
            for key,kind in [('canonical_video','canonical_video'),('marker_video','marker_video')]:
                p=row[key+'_path']
                if eligible(p): media[p]={'path':p,'kind':kind,'bytes':row[key+'_bytes'],'sha256':row[key+'_sha256']}
            for key in ['accepted','inventory']:
                p=row[key+'_path']
                if eligible(p): evidence[p]={'original_path':p,'bytes':row[key+'_bytes'],'sha256':row[key+'_sha256']}
            inventory=read(ROOT/row['inventory_path'])
            assert inventory['GT_read'] is False and inventory['query_independent'] is True
            for item in inventory['files']:
                p=rel(ROOT/item['path'])
                if eligible(p): evidence[p]={'original_path':p,'bytes':item['bytes'],'sha256':item['sha256']}
        for relative,item in {**media,**evidence}.items():
            p=ROOT/relative;assert p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],relative
        archived=None
        if evidence:
            archive=FULL/'tracking_evidence_archives'/(selection+'.tar.gz');archive.parent.mkdir(parents=True,exist_ok=True)
            # An interrupted archive is not replaced or presented as complete.
            with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
                for relative in sorted(evidence): tar.add(ROOT/relative,arcname=relative,recursive=False)
            archived={'path':rel(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':[evidence[p] for p in sorted(evidence)]}
            tracker._read_archive_members(archived)
        plan={'schema':tracker.RETIREMENT_SCHEMA,'status':'complete','selection_sha256':selection,
              'tracking_batch_barrier_sha256':sha(tp),'tracking_runtime_sha256':locked['tracking_runtime_sha256'],
              'downstream_seals':seals,'retired_files':[media[p] for p in sorted(media)],
              'lossless_evidence_archive':archived,'GT_read':False,'time':time.time(),
              'scope':'new FULL reproducible bridge media retired; generated masks retained losslessly; historical caches excluded'}
        write(journal,plan)
    # Every unlink has an exact current hash or a previously recorded retirement
    # journal; the archive was read back before the first deletion.
    rows=[(x['path'],x) for x in plan['retired_files']]
    rows += [(x['original_path'],x) for x in (plan['lossless_evidence_archive'] or {}).get('members',[])]
    for relative,item in rows:
        assert eligible(relative),relative
        p=ROOT/relative
        if p.exists():
            assert p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],relative
            p.unlink()
    write(out,plan)
    tracker._validate_retirement(tp,b,locked)
    return {'status':'pass','selection_sha256':selection,'media_files_retired':len(plan['retired_files']),
            'generated_mask_evidence_losslessly_preserved':True,'GT_read':False}

def main():
    import sys
    sys.path.insert(0,str(ROOT))
    p=argparse.ArgumentParser()
    for key in ['tracking','teacher','student','root']:p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(retire(*[ROOT/getattr(a,k) for k in ['tracking','teacher','student','root']])))
if __name__=='__main__': main()
