"""Measure both physical vote channels on one input. No target identities enter this module."""
import argparse
from time import perf_counter
from motion import (ARMS, DUEL, HERE, PREV, RESULTS, WORK, archive_npz, atomic, channels,
                    choose_scores, digest, engine, initialize, np, observations, read, torch)
from common import pin
from fit_duels import loaded
from fit import write_csv
from paired import capture
from scope183 import check_input,frame,ordinary
from source_features import native_boxes
from crossplatform.candidate_frontend import sensor_frame_box
from transfer import evaluate_sealed


@torch.inference_mode()
def extract():
    initialize()
    folder=WORK/'features'
    folder.mkdir(exist_ok=True)
    hist=PREV/'run_PRIVATE/target'
    histseal=read(hist/'SEALED.json')
    pin(folder/'LOCK.json',dict(motion=digest(HERE/'motion.py'),code=digest(HERE/'target_motion.py'),
            protocol=digest(HERE/'PROTOCOL.json'),roster=digest(hist/'SEALED.json'),target_labels=False))
    model=engine()
    roster=ordinary()
    start=perf_counter()
    all_files={}
    for t in ('drone','quad'):
        files={}
        part=folder/t
        part.mkdir(exist_ok=True)
        for name,spec in sorted(histseal['files'][t].items()):
            path=part/name
            done=path.with_suffix('.json')
            if done.exists():
                receipt=read(done)
                assert receipt['sha256']==digest(path) and receipt['prior']==spec['sha256']
            else:
                original=hist/t/name
                assert digest(original)==spec['sha256']
                old=loaded(original)
                vals,good,parsed=[],[],[]
                checked=False
                for n,idx in enumerate(old['ordinals']):
                    row=roster[t][idx]
                    check_input(row)
                    full=observations(model,frame(row),parity=not checked)
                    checked|=full.neutral is not None
                    boxes=native_boxes(full.ev)
                    physical=np.stack([sensor_frame_box(b,t,full.full.view_index) for b in boxes])
                    np.testing.assert_array_equal(physical,old['four'][n])
                    _,j=capture(model._decoder,full.ev)
                    np.testing.assert_allclose(j,old['scores'][n,0],atol=2e-6,rtol=0)
                    v,k=channels(boxes,full)
                    vals.append(v)
                    good.append(k)
                    parsed.append(full.neutral is not None)
                archive_npz(path,val=np.stack(vals),ok=np.stack(good),parsed=np.array(parsed))
                receipt=dict(sha256=digest(path),prior=spec['sha256'],n=len(good),self_replay_checked=checked)
                atomic(done,receipt)
            files[name]=receipt
            print(t,name,round(perf_counter()-start,1),flush=True)
        all_files[t]=files
    atomic(folder/'SEALED.json',dict(files=all_files,lock=digest(folder/'LOCK.json'),
               seconds=perf_counter()-start,target_labels=False,FINAL_access=False))


def score_target():
    initialize()
    folder=WORK/'features'
    spec=read(folder/'SEALED.json')
    assert spec['lock']==digest(folder/'LOCK.json')
    lock=read(WORK/'MODEL_LOCK.json')
    assert lock['motion']==digest(HERE/'motion.py') and lock['info']==digest(HERE/'PROTOCOL.json')
    hist=PREV/'run_PRIVATE/target'
    all_box,all_pick,seq,targets=[],[],[],[]
    coverage=[]
    for t,files in spec['files'].items():
        valid_counts,known,measured=[],0,[]
        for name,item in sorted(files.items()):
            f=folder/t/name
            assert digest(f)==item['sha256'] and digest(hist/t/name)==item['prior']
            obs=loaded(f)
            old=loaded(hist/t/name)
            selected=[]
            for j,v,k in zip(old['scores'][:,0],obs['val'],obs['ok']):
                pred=choose_scores(j,v,k,lock['coefficients'])
                assert tuple(pred)==ARMS
                for e in pred.values():
                    np.testing.assert_allclose(e[~k],j[~k],atol=0,rtol=0)
                    if k.any():
                        np.testing.assert_allclose(np.logaddexp.reduce(e[k].astype(float)),
                                                   np.logaddexp.reduce(j[k].astype(float)),atol=1e-10,rtol=0)
                selected.append([np.argmax(x) for x in pred.values()])
                valid_counts.append(int(k.sum()))
                if k.sum()>=2:
                    measured.append(float(np.ptp(v[k,0])))
            known+=int(obs['parsed'].sum())
            np.testing.assert_array_equal(np.array(selected)[:,0],old['choices'][:,0])
            all_box.append(old['four'])
            all_pick.extend(selected)
            seq.extend(old['ordinals'])
            targets.extend([t]*len(selected))
        coverage.append(dict(target=t,n=len(valid_counts),parsed=known,
            comparable=int((np.array(valid_counts)>=2).sum()),partial=int(((np.array(valid_counts)>=2)&(np.array(valid_counts)<4)).sum()),
            all_four=int((np.array(valid_counts)==4).sum()),median_interaction_range=float(np.median(measured)) if measured else 0.))
    write_csv(RESULTS/'coverage.csv',coverage)
    archive_npz(WORK/'predictions.npz',picks=np.array(all_pick,dtype=np.uint8),arms=np.array(ARMS),
                four=np.concatenate(all_box),ordinals=np.array(seq),targets=np.array(targets))
    atomic(WORK/'PREDICTIONS_SEAL.json',dict(file=digest(WORK/'predictions.npz'),head=digest(WORK/'MODEL_LOCK.json'),
        features=digest(folder/'SEALED.json'),evaluator=digest(DUEL/'transfer.py'),target_annotations_read=False,
        final_access=False))
    def controls(arm):
        return ['v1','permuted_interaction','category_prior','remove_interaction'] if arm=='source_interaction' else ['v1']
    evaluate_sealed(WORK,RESULTS,controls,('source_interaction','remove_interaction','category_prior'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['extract','score'])
    if parser.parse_args().stage=='extract':
        extract()
    else:
        score_target()
