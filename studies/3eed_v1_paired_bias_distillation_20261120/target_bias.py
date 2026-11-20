from bias import ARMS, DUEL, HERE, RESULTS, WORK, archive_npz, atomic, digest, initialize, loaded, np, predict, read
from common import PREV
from transfer import evaluate_sealed


def seal():
    lock=read(WORK/'LOCK.json')
    assert lock['model']==digest(WORK/'model.npz') and lock['config']==digest(HERE/'PROTOCOL.json')
    assert lock['code']==digest(HERE/'bias.py')
    model=loaded(WORK/'model.npz')
    feat=DUEL/'run_PRIVATE/target_features'
    metadata=read(feat/'SEALED.json')
    assert metadata['lock']==digest(feat/'LOCK.json') and not metadata['target_labels_read']
    history=PREV/'run_PRIVATE/target'
    proof=read(history/'SEALED.json')
    picks,boxes,indices,targets=[],[],[],[]
    for target,files in metadata['files'].items():
        for name,pin in sorted(files.items()):
            p=feat/target/name
            original=history/target/name
            assert digest(p)==pin['sha256']
            assert digest(original)==pin['previous']==proof['files'][target][name]['sha256']
            observations,old=loaded(p),loaded(original)
            seq=observations['ordinals']
            np.testing.assert_array_equal(seq,old['ordinals'])
            outputs=predict(observations['h'],observations['j'],model,lock['scales'])
            chosen=np.stack([s.argmax(-1) for s in outputs.values()],axis=-1).astype(np.uint8)
            np.testing.assert_array_equal(chosen[:,0],old['choices'][:,0])
            picks.append(chosen)
            boxes.append(old['four'])
            indices.extend(seq)
            targets.extend([target]*len(seq))
    archive_npz(WORK/'predictions.npz',arms=np.array(list(outputs)),picks=np.concatenate(picks),
                four=np.concatenate(boxes),ordinals=np.array(indices),targets=np.array(targets))
    atomic(WORK/'PREDICTIONS_SEAL.json',dict(file=digest(WORK/'predictions.npz'),head=digest(WORK/'LOCK.json'),
        code=digest(HERE/'target_bias.py'),features=digest(feat/'SEALED.json'),
        evaluator=digest(DUEL/'transfer.py'),target_annotations_read=False,final_access=False))


def controls(a):
    other=['v1']
    if a=='paired_bias':
        other+=['independent_logistic','constant_separation','broken_bias','rank_only']
    if a=='paired_bias_matched':
        other+=['independent_logistic_matched','constant_separation_matched','broken_bias_matched','rank_only']
    return other


if __name__=='__main__':
    initialize()
    seal()
    evaluate_sealed(WORK,RESULTS,controls,('paired_bias','independent_logistic','paired_bias_matched'))
