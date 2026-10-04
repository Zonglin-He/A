"""Prepare a reviewed exact Git tree for authenticated connector publication."""
import sys,time,shutil,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha

def shell(args,cwd):return subprocess.check_output(args,cwd=cwd).decode().strip()

def run(task):
    assert task in ['correction_scope','cross_domain_qualification']
    namespace='tastvg_'+task+'_v1';base=ROOT/'artifacts'/namespace
    tag='tastvg_'+('correction_scope' if task=='correction_scope' else 'cross_domain_qualification')
    pub=ROOT/'results'/tag/'2026-10-04';checkout=Path('/home/wwww/visual-grounding-public-A')
    assert read(base/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(base/'ROOT_AUDIT.json')['status']=='pass'
    assert read(base/'CPU_COMPLETION.json')['status']=='completed_pending_root_visual_publication'
    assert shell(['git','remote','get-url','origin'],checkout)=='https://github.com/Zonglin-He/A.git'
    assert not shell(['git','status','--porcelain'],checkout),'Unrelated public edits'
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=checkout,check=True)
    assert shell(['git','rev-parse','HEAD'],checkout)==shell(['git','rev-parse','origin/main'],checkout)
    if task=='correction_scope':
        stem='tastvg_correction_scope'
        files=[f'scripts/{prefix}_{stem}_v1.py' for prefix in ['prepare','run','continue','score','audit','report']]
        files += ['scripts/tastvg_correction_scope_common_v1.py','scripts/finalize_tastvg_correction_scope_cpu_v1.py',
            'protocols/tastvg_correction_scope_v1.md','docs/tastvg_correction_scope_v1/EXECUTION.md','docs/TA_CORRECTION_SCOPE_REVIEW.md']
        audit_script='audit_tastvg_correction_scope_v1.py';audit_args=[str(checkout/pub.relative_to(ROOT))]
    else:
        files=[str(f.relative_to(ROOT)) for f in (ROOT/'scripts').glob('*tastvg_cross_domain*v1.py')]
        files += ['protocols/tastvg_cross_domain_qualification_v1.md','docs/tastvg_cross_domain_qualification_v1/EXECUTION.md',
            'docs/TA_CROSS_DOMAIN_QUALIFICATION_REVIEW.md']
        audit_script='audit_tastvg_cross_domain_v1.py';audit_args=['public',str(checkout/pub.relative_to(ROOT))]
    files += ['scripts/tastvg_cpu_handoff_v1.py','scripts/prepare_tastvg_qualification_public_v1.py']
    pins=dict(read(base/'RUNTIME_LOCK.json')['pins']);pins.update(read(base/'CPU_IMPLEMENTATION_LOCK.json')['pins'])
    for folder in ['revisions','cpu_revisions']:
        for f in sorted((base/folder).glob('*.json')):pins.update(read(f)['pin_overrides'])
    dependencies={f:h for f,h in pins.items() if f.endswith('.py') and f not in files}
    for f,h in dependencies.items():
        assert sha(ROOT/f)==h,f
        if not (checkout/f).exists() or sha(checkout/f)!=h:
            # The reviewed exact dependency is code, never research data.
            assert f.startswith(('scripts/','vg_tta/','methods/','external/')),f
            files.append(f)
    files=sorted(set(files));bindings=dict(new_or_required_code={f:sha(ROOT/f) for f in files},
        existing_code_dependencies={f:h for f,h in dependencies.items() if f not in files},
        runtime_lock_sha256=sha(base/'RUNTIME_LOCK.json'),CPU_implementation_lock_sha256=sha(base/'CPU_IMPLEMENTATION_LOCK.json'),
        engineering_revisions={folder:{f.name:read(f) for f in (base/folder).glob('*.json')} for folder in ['revisions','cpu_revisions']},
        CURRENT_METHOD_unchanged=True,private_media_weights_annotations_cache_and_raw_payloads_exported=False,time=time.time())
    write(pub/'CODE_BINDING.json',bindings)
    files += [str(f.relative_to(ROOT)) for f in pub.rglob('*') if f.is_file()]
    assert all(not f.startswith(('artifacts/','data/','checkpoints/','.cache/','.codex/')) for f in files)
    for rel in files:
        f=ROOT/rel;dest=checkout/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(checkout/'scripts'/audit_script),*audit_args],cwd=checkout,check=True)
    subprocess.run(['git','add','--',*files],cwd=checkout,check=True)
    changed=shell(['git','diff','--cached','--name-only'],checkout).splitlines()
    assert set(changed)<=set(files) and changed
    entries=[]
    for rel in changed:
        blob=(checkout/rel).read_bytes();entries.append(dict(path=rel,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
            blob_sha=shell(['git','hash-object',rel],checkout),mode=shell(['git','ls-files','--stage',rel],checkout).split()[0]))
    manifest=dict(repository='Zonglin-He/A',task=task,parent_sha=shell(['git','rev-parse','HEAD'],checkout),
        base_tree=shell(['git','rev-parse','HEAD^{tree}'],checkout),tree_sha=shell(['git','write-tree'],checkout),
        files=entries,unchanged_reviewed_paths=[f for f in files if f not in changed],time=time.time())
    write(base/'PUBLICATION_MANIFEST.json',manifest)
    print('PUBLIC_TREE_READY',task,len(changed),sum(e['bytes'] for e in entries),manifest['tree_sha'],flush=True)

if __name__=='__main__':run(sys.argv[1])
