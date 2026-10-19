"""Deterministic, category-free adapter over already cached dependency tokens.

This is a research condition builder, not a change to the deployed parser.
No labels, model inference, synonym search, or per-query prompt alternatives.
"""
import copy
VERSION = 'what-role-object-relative-v1'


def rewrite(parses):
    ctx = parses['context']
    if ctx.get('eligible') or parses['old'].get('phrase') or ctx.get('reason') != 'unresolved_wh_category':
        return copy.deepcopy(parses), dict(changed=False, version=VERSION)
    tokens = ctx['tokens']
    if not tokens or tokens[0]['lemma'].lower() != 'what' or tokens[0]['deprel'] not in ('nsubj', 'obj', 'obl'):
        return copy.deepcopy(parses), dict(changed=False, version=VERSION, reason='unsupported_unknown_role')
    wh = tokens[0]; role = wh['deprel']; body = [dict(t) for t in tokens[1:] if t['text'] not in ('?', '.', '!')]
    repair = False
    if role in ('obj', 'obl'):
        # Undo question auxiliary inversion; keep do-support as emphatic do.
        roots = [t for t in body if t['head'] == 0]
        assert len(roots) == 1
        root = roots[0]; predicate = root
        if root['upos'] in ('ADV', 'ADP'):
            # Narrow cached-parser anomaly: a stranded root particle follows a
            # nominal mistag attached to the subject. Promote that adjacent
            # lexical token, without changing or inventing any source word.
            idx = body.index(root)
            assert idx > 0
            prev = body[idx - 1]
            subjects = [t for t in body if t['deprel'] == 'nsubj' and t['head'] == root['id']]
            assert len(subjects) == 1 and prev['upos'] == 'NOUN' and prev['head'] == subjects[0]['id']
            predicate = prev; repair = True
        assert predicate['id'] != wh['id']
        subjects = [t for t in body if t['deprel'] == 'nsubj' and t['head'] == root['id']]
        assert len(subjects) == 1
        subject = subjects[0]
        inverted = [t for t in body if t['upos'] == 'AUX' and t['id'] < subject['id'] and t['head'] == root['id']]
        if inverted:
            # Carry immediately following pre-subject negation with auxiliary.
            moving = []
            for t in body:
                if t in inverted or (t['id'] < subject['id'] and t['lemma'].lower() in ('not', "n't") and t['head'] in {x['id'] for x in inverted}):
                    moving.append(t)
            body = [t for t in body if t not in moving]
            at = body.index(predicate); body[at:at] = moving
    words = [t['text'].lower() for t in body]
    text = 'the object that ' + ' '.join(words) + '.'
    # Every source content word is retained once; only wh/punctuation replaced.
    assert sorted(t['text'].lower() for t in body) == sorted(t['text'].lower() for t in tokens[1:] if t['text'] not in ('?', '.', '!'))
    out = copy.deepcopy(parses)
    out['context'] = dict(eligible=True, reason=VERSION, context=text, span=[4, 10], entity='object', tokens=copy.deepcopy(tokens))
    assert text[4:10] == 'object'
    return out, dict(changed=True, version=VERSION, role=role, cached_particle_root_repair=repair, target_span=[4, 10], target_entity='object', preserves_source_words=True)
