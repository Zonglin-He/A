"""Review-only prose correction; preserve original report, figures and scores."""
import sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_native_instance_v1 as original
assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()=='d0c15175cfd5da31d4ce13736204608cd1c01b6a659b8f70927ea6aa7fb85e8d'
rows,summary,indexed,cases=original._load_scored_rows()
text=original._report_text(rows,summary,indexed,cases,original._load_cost_rows(cases,indexed))
old='That attempt produced no token output, and its generation-call count is unknown.'
new='No accepted teacher output was saved for that attempt; any partial generated tokens and its generation-call count are unknown.'
assert text.count(old)==1
text=text.replace(old,new)
text+='\n\n## Additional post-seal coverage and engineering readback\n\nSee [filtered instance-pool coverage](STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md) for the complete eight-source candidate geometry diagnostic. It is not raw detector recall or a GT-selected reference for fitting. The direct teacher baseline uses clip-relative R1 seconds mapped to the first source physical frame. A pre-publication reader initially mixed this with absolute source timestamps and stopped; its failed output and original source were retained, and the corrected clock reader leaves every teacher selection, sparse reference and student prediction unchanged. Restricted NumPy dtype and document-level roster-field dispatch corrections likewise affect CPU readback only.\n'
p=ROOT/'docs/STVG_NATIVE_INSTANCE_V1_REVIEWED.md'
with p.open('x') as f:f.write(text)
print(p)
