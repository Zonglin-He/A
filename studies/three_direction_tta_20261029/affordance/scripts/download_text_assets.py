"""Fetch tokenizer/config only, with pinned revision and observed content hashes."""
import pathlib,json,hashlib,requests
ROOT=pathlib.Path(__file__).resolve().parents[1];spec=json.loads((ROOT/'protocols/text_assets.json').read_text());out=ROOT/'assets/robbert-v2-dutch-base';out.mkdir(parents=True,exist_ok=True)
for name,digest in spec['files_sha256'].items():
 r=requests.get(f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}",timeout=90);r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==digest,name;(out/name).write_bytes(r.content)
print('Pinned text configuration/tokenizer hashes verified. No model weights fetched.')
