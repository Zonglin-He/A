"""Root-authorized score-blind inventory supplement of actual cached V inputs."""
from pathlib import Path
import json,hashlib
import inventory49 as inv
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/round49/subject_inputs_attempt001'
def main():
    assert not OUT.exists();OUT.mkdir()
    rows=json.loads((ROOT/'research/round24/input/ALL336_PRIVATE.json').read_text());assert len(rows)==336
    result=[]
    for r in rows:
        b=ROOT/'crossplatform'/('round24' if r['stage']=='M' else 'round23');suffix='M64' if r['stage']=='M' else '272'
        path=b/('text'+suffix+'_attempt001')/'cases_PRIVATE'/(r['id']+'.json');text=inv.cached(path)
        result.append({'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'stage':r['stage'],
            'caption_PRIVATE':r['caption'],'full_caption_sha256':hashlib.sha256(r['caption'].encode()).hexdigest(),
            'subject_query_PRIVATE':text['subject_query'],'subject_query_sha256':hashlib.sha256(text['subject_query'].encode()).hexdigest(),
            'actual_visual_input_source_PRIVATE':str(path.relative_to(ROOT)),'actual_visual_input_field':'subject_query',
            'cached_text_file_sha256':inv.sha(path),'visual_evidence_has_no_full_description_relationship_claim':True})
    def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    write(OUT/'TEXT_INPUTS_PRIVATE.json',result)
    write(OUT/'SOURCE_BINDINGS_PRIVATE.json',inv.BINDINGS)
    write(OUT/'SEALED.json',{'state':'ACTUAL_VISUAL_SUBJECT_INPUTS_NO_SCORES_NO_BLIND_ANSWERS',
        'n':336,'code_sha256':inv.sha(__file__),'files':{p.name:inv.sha(p) for p in OUT.iterdir() if p.is_file()}})
    print('SUBJECT_INPUTS49_SEALED',len(result))
if __name__=='__main__':main()
