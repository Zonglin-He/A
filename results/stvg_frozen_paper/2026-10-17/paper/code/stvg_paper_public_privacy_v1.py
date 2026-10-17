"""Original privacy checks with one explicit user-authorized paper .tex path."""
from pathlib import Path
import inspect
from scripts import publish_stvg_soft_ep_v1 as original
ORIGINAL_VERIFY_SOURCE=inspect.getsource(original.verify_privacy)
TEX='paper/stvg_t2_b_episodic_20261017.tex'
def verify_privacy(paths):
 tex={p for p in paths if Path(p).suffix=='.tex'};assert tex<={TEX},'only the single new paper source is authorized'
 for rel in tex:
  raw=(original.ROOT/rel).read_text();assert not any(x in raw for x in ('\\write18','\\input{','\\include{','\\openout','BEGIN PRIVATE KEY','ghp_','sk-proj-'))
 ns=dict(original.__dict__);ns['ALLOWED_SUFFIXES']=set(original.ALLOWED_SUFFIXES)|{'.tex'};exec(compile(ORIGINAL_VERIFY_SOURCE,__file__,'exec'),ns);ns['verify_privacy'](set(paths))
