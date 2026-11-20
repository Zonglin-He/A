"""A camera-aligned reflection and span-preserving word interventions."""
from dataclasses import dataclass
import re

import numpy as np

HORIZONTAL = re.compile(r'\b(?:left|right)(?:most|ward(?:s)?)?\b', re.I)
UNSUPPORTED = re.compile(r'\b(?:east|west|north|south|clockwise|counterclockwise|anticlockwise)\b|\bright\s+(?:in front|next|now|behind|up|at that moment)\b|\bleft\s+(?:over|behind)\b', re.I)


def exchange(word):
    n = 4 if word.lower().startswith('left') else 5
    prefix = 'right' if n == 4 else 'left'
    if word[:n].isupper():
        prefix = prefix.upper()
    elif word[0].isupper():
        prefix = prefix.capitalize()
    return prefix + word[n:]


def alter(text, spans, positions):
    edits = [(m.start(), m.end(), exchange(m.group())) for m in HORIZONTAL.finditer(text) if m.start() in positions]
    def boundary(i):
        if any(lo < i < hi for lo, hi, _ in edits):
            raise ValueError('Role boundary inside a direction word')
        return i + sum(len(word)-(hi-lo) for lo, hi, word in edits if hi <= i)
    result = text
    for lo, hi, word in reversed(edits):
        result = result[:lo] + word + result[hi:]
    return result, [(boundary(lo), boundary(hi)) for lo, hi in spans]


def conditions(text, graph, document, active):
    from mechanisms import distinct_entity_relation
    if not distinct_entity_relation(graph, active) or UNSUPPORTED.search(text):
        return None
    edges = [e for e in graph['relations'] if 'T' in (e['subject'], e['reference'])]
    if len(edges) != 1:
        return None
    edge = edges[0]
    entities = {e['role']:e for e in graph['entities']}
    role = edge['reference'] if edge['subject']=='T' else edge['subject']
    word = next((w for w in document['sentences'][edge['sentence']]['words']
                 if w['id']==edge['governor_word']), None)
    if not word or word.get('start_char') is None or word.get('end_char') is None:
        return None
    lo, hi = word['start_char'], word['end_char']
    if text[lo:hi].upper() != edge['predicate'] or edge['predicate'] not in ('LEFT','RIGHT'):
        return None
    positions = {m.start() for m in HORIZONTAL.finditer(text)}
    if lo not in positions:
        return None
    spans = [entities['T']['span'], entities[role]['span'], (lo, hi)]
    try:
        return [alter(text, spans, x) for x in (set(),positions,{lo},positions-{lo})]
    except ValueError:
        return None


@dataclass(frozen=True)
class Mirror:
    origin: np.ndarray
    normal: np.ndarray

    @classmethod
    def camera(cls, matrix):
        rotation, translation = matrix[:3,:3], matrix[:3,3]
        right = rotation[0].copy()
        right[2] = 0.
        norm = np.linalg.norm(right)
        if norm < 1e-8:
            return None
        return cls(-rotation.T @ translation, right/norm)

    def points(self, points):
        delta = np.asarray(points, dtype=np.float64)-self.origin
        return np.asarray(points)-2*(delta@self.normal)[...,None]*self.normal

    def boxes(self, boxes):
        out = np.array(boxes,dtype=np.float64,copy=True)
        out[...,:3] = self.points(out[...,:3])
        t = out[...,6]
        axis = np.stack((np.cos(t),np.sin(t),np.zeros_like(t)),axis=-1)
        axis = axis - 2*(axis@self.normal)[...,None]*self.normal
        out[...,6] = (np.arctan2(axis[...,1],axis[...,0])+np.pi/2)%np.pi-np.pi/2
        return out
