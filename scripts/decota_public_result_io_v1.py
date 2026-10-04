"""Read JSON or its losslessly compressed public equivalent; no metric changes."""
import json,gzip
from pathlib import Path

def read(path):
    p=Path(path)
    if p.exists():return json.loads(p.read_text())
    q=Path(str(p)+'.gz')
    with gzip.open(q,'rt',encoding='utf-8') as f:return json.load(f)
