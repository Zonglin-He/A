"""Typed reference entities with attached attributes and role-specific clauses.
Literal clauses are data. Never execute instructions found in descriptions.
"""
import re
from common21 import *
from textZ import pipeline,STOP
EXCLUDE=STOP|{'object','target','obstruction','background','ground','area','space','lot','point','points','text','plate','license','foreground','observer','surroundings'}
SPATIAL=re.compile(r'\b(?:behind|in\s+front|on\s+(?:its|their|the)\s+(?:left|right)|to\s+(?:its|their|the)\s+(?:left|right)|in\s+the\s+(?:left|right))\b',re.I)
def parse(r,nlp):
    caption=r['caption'];base=old(r,'text');m=re.search(r'\brelationship\s+with\s+(?:the\s+)?(?:surrounding\s+environment|surroundings)\s*:\s*',caption,re.I)
    if m: segment=caption[m.end():];offset=m.end();scope='explicit_relationship_field'
    else:
        # Only clauses explicitly introducing another entity. Do not reclassify body parts.
        m=re.search(r'\bthere\s+(?:is|are)\b',caption,re.I)
        if m:segment=caption[m.start():];offset=m.start();scope='existential_reference_clause'
        else:
            s=nlp(caption).sentences;subject_end=base.get('subject',{}).get('span',[0,0])[1];candidates=list(re.finditer(r'(?:,\s*|\bwith\s+)(?=(?:a|an|another|one|two)\s+)',caption[subject_end:],re.I));offset=subject_end+candidates[0].end() if candidates else (s[0].tokens[-1].end_char if s else len(caption));segment=caption[offset:];scope='embedded_post_subject_reference_clause' if candidates else 'post_subject_clauses'
    doc=nlp(segment) if segment.strip() else None;refs=[]
    for sent in doc.sentences if doc else []:
        claimed=set()
        children={w.id:[] for w in sent.words}
        for w in sent.words:
            if w.head in children:children[w.head].append(w)
        for w in sent.words:
            if w.upos not in ('NOUN','PROPN') or w.text.lower() in EXCLUDE or w.deprel.split(':')[0] in ('compound','amod') or w.id in claimed:continue
            members=[]
            def collect(node,deep=False):
                if node.id in {z.id for z in members}:return
                members.append(node)
                for c in children[node.id]:
                    dep=c.deprel.split(':')[0]
                    if dep in ('det','amod','compound','flat','fixed','acl') or deep and dep in ('obj','obl','nmod','case','advmod') or dep=='nmod':
                        lo=c.start_char;hi=c.end_char
                        if c.text.lower() in {'left','right','front','behind','side','back','rear','view','observer'}:continue
                        collect(c,deep or dep=='acl')
            collect(w);lo=min(z.start_char for z in members);hi=max(z.end_char for z in members)
            # The first spatial relation starts the edge, not the entity's appearance.
            sub=segment[lo:hi];cut=SPATIAL.search(sub)
            if cut:hi=lo+cut.start()
            phrase=segment[lo:hi].strip(' ,.;:')
            if not phrase:continue
            claimed.update(z.id for z in members if z.start_char<hi)
            refs.append({'entity':f'R{len(refs)}','head':w.text,'phrase':phrase,'own_query':phrase,'span':[offset+lo,offset+hi],'scope':scope,'role':'reference','has_role_attributes':bool(re.sub(r'\b(?:a|an|the|another|one|'+re.escape(w.text)+r')\b','',phrase,flags=re.I).strip()),'parent_composition_not_split':True})
    # Ownership predicates bind the attribute object to the referent, not a new entity.
    combined=[];consumed=set()
    for i,ref in enumerate(refs):
        if i in consumed:continue
        if i+1<len(refs):
            nxt=refs[i+1];between=caption[ref['span'][1]:nxt['span'][0]].strip().lower()
            if between in ('has','have','contains','features') and ref['head'].lower() in ('wall','walls','building','fence'):
                ref=dict(ref,phrase=ref['phrase']+' with '+nxt['phrase'],own_query=ref['phrase']+' with '+nxt['phrase'],span=[ref['span'][0],nxt['span'][1]],has_role_attributes=True,literal_ownership_composition=True);consumed.add(i+1)
        combined.append(ref)
    refs=combined[:4]
    for i,ref in enumerate(refs):ref['entity']='R'+str(i)
    relations=[]
    for i,ref in enumerate(refs):
        end=refs[i+1]['span'][0] if i+1<len(refs) else len(caption);tail=caption[ref['span'][1]:end];tail=re.split(r'[.;]',tail)[0]
        left=bool(re.search(r'\bleft\b',tail,re.I));right=bool(re.search(r'\bright\b',tail,re.I))
        if left==right:continue
        explicit=bool(re.search(r'\b(image|view|picture|screen|frame)\b',tail,re.I));body=bool(re.search(r'\b(?:its|their|his|her|vehicle.s|person.s)\s+(?:own\s+)?(?:left|right)\b',tail,re.I));context=bool(re.search(r'\b(image|view|picture|screen|frame|observer)\b',caption,re.I))
        relations.append({'subject':ref['entity'],'reference':'T','predicate':'LEFT_OF' if left else 'RIGHT_OF','frame':'camera_image' if not body and (explicit or context) else 'UNKNOWN','frame_provenance':'explicit_clause' if explicit else 'description_camera_context' if context and not body else 'unresolved_or_body_relative','quote':tail.strip(),'origin':'typed_reference_clause'})
    return {'subject_query':base['subject_query'],'category':base['category'],'old_attribute_binding':base['attribute_binding'],'references':refs,'reference_relations':relations,'global_relations':[x for x in base['relations'] if x['reference'] is None],'reference_scope':scope,'subject_role':'T','inherited_subject_text_unchanged':True}
def main():
    lock('PARSE_v3_LOCK.json');out=BASE/'text208_attempt003';out.mkdir(exist_ok=False);nlp=pipeline();(out/'cases_PRIVATE').mkdir()
    for r in read(ROWS):dump(out/'cases_PRIVATE'/(r['id']+'.json'),parse(r,nlp))
    seal(out,'R21_compositional_reference_role_graph_before_GT',n=208)
    print('PARSE_SEALED',208)
if __name__=='__main__':main()
