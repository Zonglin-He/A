"""O2 CPU streams: reset per regime, emit before the same sparse O1 update."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.decota_matrix_common_v1 import read,write,load,save,sha
from scripts.run_tastvg_regime_online_capture_v1 import OUT,verify
from vg_tta.tastvg_sparse_online_v1 import arrive,score,update,source_state
from methods.decota_final_simplified_v1.tensors import state_hash


def run():
    tick=time.monotonic();torch.set_num_threads(4);p=verify();bar=read(OUT/'CAPTURE_BARRIER.json');teacher=read(OUT/'TEACHER_ONLINE_BARRIER.json')
    assert len(teacher['files'])==20 and not (OUT/'teacher/full').exists()
    write(OUT/'ONLINE_LOCK.json',dict(pins={str(Path(__file__)):sha(__file__),str(ROOT/'vg_tta/tastvg_sparse_online_v1.py'):sha(ROOT/'vg_tta/tastvg_sparse_online_v1.py')},teacher_barrier_sha256=sha(OUT/'TEACHER_ONLINE_BARRIER.json'),lr=p['lr'],zero_state_sha256=state_hash(source_state())))
    previous=None;previous_record=None;access=[]
    for r in p['rows']:
        if r['stream_position']==1:previous=None;previous_record=None
        pos=r['position'];rel=f'capture/{pos:03}.pt';assert sha(OUT/rel)==bar['files'][rel];x=load(OUT/rel);state=arrive(previous);before=state_hash(state)
        if previous is not None:assert before==state_hash(previous)
        assert int(score(x['base_score'],x['phi'],source_state()).argmax())==0
        arriving_scores=score(x['base_score'],x['phi'],state);before_selected=int(arriving_scores.argmax());teacher_rel=None;diagnostic=None
        if r['expert']:
            teacher_rel=f'teacher/online/{pos:03}.pt';assert teacher_rel in teacher['files'] and sha(OUT/teacher_rel)==teacher['files'][teacher_rel]
            e=load(OUT/teacher_rel);assert e['position']==pos and e['phase']=='online';access.append(pos)
            selected=e['selected'];budgeted=selected
            # Current output uses expert directly; the following state is future-only.
            after,diagnostic=update(x['base_score'],x['phi'],state,e['scores'],p['lr'])
        else:selected=before_selected;budgeted=0;after=arrive(state)
        result=dict(position=pos,stream_position=r['stream_position'],parent=r['ordinal'],condition=r['condition'],expert=r['expert'],selected={'Frozen':0,'Budgeted Rerank':budgeted,'Online Slow-Fast':selected},before_selected=before_selected,
            base_score=x['base_score'],arrival_scores=arriving_scores,arrival_state=state,after_state=after,arrival_hash=before,after_hash=state_hash(after),previous_record_sha256=previous_record,
            teacher_read=teacher_rel,diagnostics=diagnostic,output_before_update=True,GT_read=False)
        path=OUT/'online'/f'{pos:03}.pt';save(path,result);write(path.with_suffix('.json'),dict(sha256=sha(path)));previous_record=sha(path);previous=after
    assert access==p['expert_positions'];verify()
    write(OUT/'ONLINE_BARRIER.json',dict(files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'online').glob('*.pt')},positions=80,expert_reads=access,expert_read_count=20,nonexpert_teacher_reads=0,updates=20,final_state_sha256=state_hash(previous),seconds=time.monotonic()-tick,GT_read=False,time=time.time()))
    print('ONLINE SEALED',80,'positions,',len(access),'expert reads; no GT or nonexpert expert reads')


def full():
    p=verify();ob=read(OUT/'ONLINE_BARRIER.json');tb=read(OUT/'TEACHER_FULL_BARRIER.json');assert tb['online_barrier_sha256']==sha(OUT/'ONLINE_BARRIER.json')
    selected=[]
    for r in p['rows']:
        pos=r['position'];rel=f'online/{pos:03}.pt';assert sha(OUT/rel)==ob['files'][rel];x=load(OUT/rel)
        phase='online' if r['expert'] else 'full';rel=f'teacher/{phase}/{pos:03}.pt';bar=read(OUT/f'TEACHER_{phase.upper()}_BARRIER.json');assert sha(OUT/rel)==bar['files'][rel];e=load(OUT/rel)
        selected.append(dict(position=pos,stream_position=r['stream_position'],parent=r['ordinal'],condition=r['condition'],expert=r['expert'],selected={**x['selected'],'Full Rerank':e['selected']}))
    write(OUT/'SEALED_SELECTIONS.json',selected)
    write(OUT/'PREDICTION_BARRIER.json',dict(selection_sha256=sha(OUT/'SEALED_SELECTIONS.json'),online_barrier_sha256=sha(OUT/'ONLINE_BARRIER.json'),full_teacher_barrier_sha256=sha(OUT/'TEACHER_FULL_BARRIER.json'),cells=80,arms=4,GT_read=False,time=time.time()))
    print('FOUR ARM OUTPUTS SEALED')

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['online','full']);x=a.parse_args();run() if x.stage=='online' else full()
