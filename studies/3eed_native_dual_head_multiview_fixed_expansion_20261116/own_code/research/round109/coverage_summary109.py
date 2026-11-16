"""Postseal metadata-only coverage artifact, no model or new task scoring."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round109';B=R/'crossplatform/round109';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'NATIVE_AUDIT.json')['n']==1536;groups=[]
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rows=[rd(p) for p in sorted((B/name/'records_PRIVATE').glob('*.json'))];assert len(rows)==256;src,_,tgt=name.split('_');groups.append({'source':src,'target':tgt,'n':len(rows),'protected_alias_fallback':sum(r['protected_alias_fallback'] for r in rows),'families':{k:{v+'_available':sum(r.get('families',{}).get(k,{}).get(v+'_available',False) for r in rows) for v in ['B','C']} for k in ['real','randomA','wrong']}})
 (H/'COVERAGE_MODEL_SUMMARY.json').write_text(json.dumps({'groups':groups,'all1536_retained':True,'metadata_only_from_sealed_records':True},indent=2)+'\n');print('COVERAGE109_METADATA',len(groups))
if __name__=='__main__':main()
