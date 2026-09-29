"""Metadata-only full official-test roster and all conditions/orders; no labels or scores."""
import sys,time,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_full_b1_v1'

def run():
    parent=ROOT/'artifacts/decota_final_freeze_v1/LOCK.json';p=read(parent);exfiles=[ROOT/'artifacts/tastvg_evidence_vulnerability_v1/full64_v1/LOCK.json',ROOT/'artifacts/tastvg_corruption_c0c1_v1/LOCK.json'];excluded={r['source'] for f in exfiles for r in read(f)['rows'] if r['cohort']=='vidstg_test'}
    # Subsequent same-route rosters must be subsets of the exclusion union.
    for f in ['artifacts/tastvg_temporal_fourarm_v1/LOCK.json','artifacts/tastvg_schedule_j01_v1/LOCK.json']:
        assert {r['source'] for r in read(ROOT/f)['rows']}<=excluded
    full=p['rows']['vidstg_test'];raw=[r for r in full if r['input']['source'] not in excluded];sources=sorted({r['input']['source'] for r in raw});assert len(raw)==9411 and len(sources)==670
    rows=[]
    for i,r in enumerate(raw):
        q=r['input'];assert not r['input_unavailable'];assert Path(q['video_path']).is_file()
        rows.append(dict(ordinal=i,key=r['key'],source=q['source'],input=q,frame_ids=q['frame_ids'],query_type=r['query_type']))
    order={}
    for j in range(1,4):
        order[f'order{j}']=[r['ordinal'] for r in sorted(rows,key=lambda r:(hashlib.sha256(f'B1-source-order-20260929|{j}|{r["source"]}'.encode()).hexdigest(),r['source'],hashlib.sha256(f'B1-query-order-20260929|{r["key"]}'.encode()).hexdigest(),r['key']))]
    needed=sorted({q for seq in order.values() for i,q in enumerate(seq) if i%4==0})
    conditions=['clean']+[f'{f}_{s}' for f in ['frame_drop','frame_freeze','motion_blur','occlusion','exposure'] for s in [1,5,10]]
    method=ROOT/'methods/tastvg_dual_evidence_j0_v1';freeze=read(method/'FREEZE_J01.json');assert sha(method/'method.py')==freeze['method_code_sha256'] and sha(method/'config.json')==freeze['config_sha256']
    receipt=dict(status='cohort_and_order_locked_before_new_predictions',sources=670,queries=9411,official_sources=732,official_queries=10303,excluded_sources=62,excluded_queries=892,num_orders=3,conditions=conditions,online_arrivals=9411*3*16,scheduled_per_order=len(range(0,9411,4)),unique_expert_queries=len(needed),expert_query_conditions=len(needed)*16,globally_fresh=False,current_route_development_sources_excluded=True,all_retained_source_queries=True,query_within_source_hash_fixed=True,expert_every_four_query_arrivals=True,source_contiguous=True,GT_used_for_selection=False,history='User approved official test with historical project exposure. Train not used with Vid-trained checkpoint.',parent_sha256=sha(parent),excluded_roster_hashes={str(f.relative_to(ROOT)):sha(f) for f in exfiles},freeze_sha256=sha(method/'FREEZE_J01.json'),time=time.time())
    write(OUT/'ROSTER_LOCK.json',dict(**receipt,rows=rows,orders=order,expert_needed=needed,excluded=sorted(excluded)))
    write(OUT/'COHORT.json',receipt)
    write(OUT/'ORDERS.json',dict(orders={k:[f'Q{x+1:05}' for x in v] for k,v in order.items()},expert_positions='arrival modulo4 equals0',source_order_rule='SHA256 B1-source-order-20260929|j|source',within_source_rule='SHA256 B1-query-order-20260929|key'))
    print(receipt)

if __name__=='__main__':run()
