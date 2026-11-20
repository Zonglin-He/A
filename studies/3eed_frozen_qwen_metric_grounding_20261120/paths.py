import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIV = HERE/'private'
OUT = HERE/'results'
SPEC = HERE/'PROTOCOL.json'
PLATFORMS = ('waymo','drone','quad')

def read(path):
    return json.loads(Path(path).read_bytes())

def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(path)

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def key(*args):
    return hashlib.sha256(json.dumps(args, separators=(',',':')).encode()).hexdigest()[:24]
