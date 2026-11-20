"""Research artifact IO. Inference consumes only the question roster and allowed observations."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIVATE = HERE/'private'
OUT = HERE/'results'
SPEC = HERE/'PROTOCOL.json'
PLATFORMS = ('waymo', 'drone', 'quad')

def read(path):
    return json.loads(Path(path).read_bytes())

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def key(*items):
    return hashlib.sha256(json.dumps(items, separators=(',',':')).encode()).hexdigest()[:24]
