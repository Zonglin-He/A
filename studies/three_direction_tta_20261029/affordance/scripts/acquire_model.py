"""Acquire exact source model members without downloading redundant optimizer/encoder members."""
import pathlib,requests,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];parts=ROOT/'downloads/checkpoints/best_unseen_aff.pt.parts2m';parts.mkdir(parents=True,exist_ok=True);p=parts/'0';n=2097152
if not p.exists() or p.stat().st_size!=n:
 r=requests.get('https://drive.usercontent.google.com/download?id=1yOV4KdRVWXRjXlXbYZ4_H9peoZIIKc0Y&export=download&confirm=t',headers={'Range':f'bytes=0-{n-1}'},timeout=(20,60));r.raise_for_status();assert r.status_code==206 and len(r.content)==n and r.headers.get('Content-Range')==f'bytes 0-{n-1}/1208734804';p.write_bytes(r.content)
for args in [['scripts/check_zip_duplicates.py'],['scripts/recover_weights.py','--execute'],['scripts/validate_checkpoint.py','--checkpoint','downloads/checkpoints/best_unseen_aff.model_only_crc_recovered.pt','--output','checkpoint_derived_validation.json']]:subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
