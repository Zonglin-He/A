"""Explicit user-authorized, exact-scope retirement of bulky obsolete assets.

Preserves configurations, scientific results and reports. This is not a generic
garbage collector; its immutable manifest records intentional replay dependencies.
"""
import argparse, hashlib, json, os, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/storage_cleanup_20260928_route_switch'

def files():
    specs = [
        ('data/desta3d_v3_source/hc1_train', {'.mp4', '.mkv'}, 'Cancelled full-source training extraction; HC1 original split archive retained'),
        ('data/corruption_inputs_res224_v1/objects', {'.bin'}, 'Rebuildable old corruption pixel cache; recipes and original media retained'),
        ('downloads/dataset_intake_20260910/hcstvg/datasets/HC-STVG2/HC-STVG/VIdeo', {'.zip'}, 'Unneeded complete HC2 bulk intake; metadata, extracted experiment clips and verified 36 HC1 repair clips retained'),
    ]
    for model in ['Qwen3-VL-32B-Instruct-GGUF', 'Qwen3-VL-8B-Instruct', 'Qwen3-VL-4B-Instruct', 'UniVG-R1', 'PR1-Qwen2-VL-2B-Grounding', 'vilt-b32-mlm']:
        specs.append((f'checkpoints/{model}', {'.safetensors', '.bin', '.gguf'}, 'Old paused route weights; not needed by retained PTD4B or CURRENT; official metadata/configs retained for redownload'))
    for audit in ['.audit_ranges', '.audit_tail_ranges', '.audit_final_ranges']:
        specs.append((f'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/{audit}', {'.part'}, 'Redundant download range audit bytes; final official PTD safetensors and audit metadata retained'))
    result = []
    for folder, suffixes, reason in specs:
        for p in sorted((ROOT / folder).rglob('*')):
            if p.is_file() and not p.is_symlink() and p.suffix in suffixes:
                s = p.stat()
                result.append(dict(path=str(p.relative_to(ROOT)), bytes=s.st_size, allocated=s.st_blocks * 512,
                    dev=s.st_dev, inode=s.st_ino, mtime_ns=s.st_mtime_ns, reason=reason))
    return result

def main(execute):
    OUT.mkdir(exist_ok=True, parents=True)
    manifest = OUT / 'DELETE_MANIFEST.json'
    if not execute:
        assert not manifest.exists()
        rows = files()
        by_reason = {}
        for r in rows: by_reason[r['reason']] = by_reason.get(r['reason'], 0) + r['bytes']
        obj = dict(time=time.time(), explicit_user_authorization='Stop current model training and delete unneeded checkpoints, model weights and data',
            file_count=len(rows), logical_bytes=sum(r['bytes'] for r in rows), categories=by_reason, files=rows,
            hash_policy='stat/inode identity checked immediately before deletion; no claim of new full SHA for bulk retired files; old download/media integrity records retained',
            preserved=['all artifacts and reports/predictions', 'all code/protocols', 'CURRENT and its weights', 'PTD4B final safetensors',
                       'B1_FIXED_FINAL and v3 user-stop checkpoint', 'source PANEL16 media', 'HC1/Vid original source archives',
                       'HC2 annotations and extracted diagnostic media', 'HC1 36 verified repair clips', 'STVG-R1 and TaRO specialist weights'])
        manifest.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k:v for k,v in obj.items() if k!='files'}, ensure_ascii=False));return
    assert not (OUT/'COMPLETE.json').exists()
    rows = json.loads(manifest.read_text())['files']; targets = {(r['dev'],r['inode']) for r in rows}
    opened=[]; skipped_processes=0
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    s=fd.stat()
                    if (s.st_dev,s.st_ino) in targets: opened.append(str(fd))
                except (OSError,PermissionError): pass
        except (OSError,PermissionError): skipped_processes+=1
    assert not opened, opened
    # Exact newly-needed source inputs must remain outside deletion scope.
    panel=json.loads((ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2/INPUTS.json').read_text())
    deleted={str(ROOT/r['path']) for r in rows}
    for row in panel:
        p=Path(row['input']['video_path']);assert p.is_file() and str(p) not in deleted
    assert (ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors').stat().st_size==8880014376
    before=shutil.disk_usage(ROOT).free; total=0
    with (OUT/'DELETED.jsonl').open('x') as log:
        for row in rows:
            p=ROOT/row['path'];s=p.stat()
            assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['dev'],row['inode'],row['bytes'],row['mtime_ns']),str(p)
            p.unlink();total+=row['bytes'];log.write(json.dumps({'path':row['path'],'bytes':row['bytes'],'time':time.time()})+'\n')
        log.flush();os.fsync(log.fileno())
    after=shutil.disk_usage(ROOT).free
    result=dict(status='completed',time=time.time(),files_deleted=len(rows),logical_bytes_deleted=total,free_before=before,free_after=after,
        measured_free_increase=after-before,open_fd_matches=opened,proc_scan_permission_limited_processes=skipped_processes,
        source_panel16_inputs_preserved=True,research_artifacts_deleted=False,
        reproducibility='Historical reports remain valid; exact reruns requiring retired weights/media/cache require explicit redownload or regeneration. No deleted data is represented as still present.')
    (OUT/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)
