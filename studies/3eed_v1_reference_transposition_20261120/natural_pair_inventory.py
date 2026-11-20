"""Count opportunities for same-frame natural reference-only text swaps without reading identities."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import re

from registry_swap import OUT, ordinary, save

REL = re.compile(r'\b(?:left|right)(?:\s+side)?\s+of\b', re.I)
WORDS = re.compile(r"\w+|[^\w\s]")


def normalized(text):
    return tuple(WORDS.findall(text.casefold()))


def run():
    result={}
    for target,rows in ordinary().items():
        groups=defaultdict(set)
        total=sum(1 for r in rows if REL.search(r['caption']))
        for r in rows:
            text=r['caption']
            frame=(str(Path(r['pointpath']).resolve()),int(r['suffix'] or 0))
            for m in REL.finditer(text):
                groups[frame,normalized(text[:m.end()])].add(normalized(text[m.end():]))
        options=Counter()
        same_region=0
        for (frame,_),suffixes in groups.items():
            for a,b in combinations(sorted(suffixes),2):
                options[frame]+=1
                # An identical first phrase ending at punctuation and naming a camera region
                # cannot provide a substitution of two different concrete reference entities.
                def first_phrase(tokens):
                    stop=next((i for i,w in enumerate(tokens) if w in (',',';','.')),len(tokens))
                    return tokens[:stop]
                left,right=first_phrase(a),first_phrase(b)
                if left==right and {'center','view'}<=set(left):
                    same_region+=1
        result[target]=dict(rows=len(rows),containing_direction=total,eligible_frames=len(options),
                           matched_prefix_pairs=sum(options.values()),identical_view_center_mentions=same_region,
                           remaining_possible_concrete_swaps=sum(options.values())-same_region)
    save(OUT/'natural_pair_inventory.json',dict(rule='Same physical file, camera and complete text prefix through an explicit left/right of. Distinct normalized suffixes. This is a necessary condition only for a reference change confined to the following phrase.',platforms=result,labels=False,FINAL_access=False))


if __name__=='__main__':
    run()
