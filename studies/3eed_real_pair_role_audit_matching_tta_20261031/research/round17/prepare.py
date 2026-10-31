"""Metadata/content-hash split: old A64, newly reserved B32. No task labels."""
from pathlib import Path
import sys,collections,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round17'))
from common import EXP,read,dump,sha
def main():
 d=EXP.parent/'input';d.mkdir(exist_ok=False)
 old=ROOT/'research/round15/input';A=read(old/'FRESH64_GPU_INPUTS.json');oldcases={c['id']:c for c in read(old/'FORMAL_CASE204_PRIVATE.json')};fi=read(old/'G_FRAME_INDEX_PRIVATE.json');inv=read(old/'DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];full={r['id']:r for r in read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json')}
 used=read(old/'SOURCE96_GPU_INPUTS.json')+read(old/'REGRESSION140_GPU_INPUTS.json');usedids={r['id'] for r in used};byid={r['canonical_id_PRIVATE']:r for r in inv};up={byid[r['id']]['decoded_pixel_sha256'] for r in used if r['id'] in byid};ux={byid[r['id']]['canonical_XYZ_sha256'] for r in used if r['id'] in byid};B=[]
 for platform in ('drone','quad'):
  scenes=sorted({r['scene'] for r in A if r['platform']==platform});quota=16//len(scenes)
  for scene in scenes:
   candidates=[r for r in inv if r['platform']==platform and r['scene_PRIVATE']==scene and r['finite_XYZ'] and not r['component_touches_previous_recent_input'] and r['canonical_id_PRIVATE'] not in usedids]
   candidates.sort(key=lambda r:hashlib.sha256(('R17_B_content_holdout_v1|'+r['pair_group_sha256']).encode()).hexdigest());chosen=[]
   for r in candidates:
    if r['decoded_pixel_sha256'] in up or r['canonical_XYZ_sha256'] in ux:continue
    chosen.append(full[r['canonical_id_PRIVATE']]);up.add(r['decoded_pixel_sha256']);ux.add(r['canonical_XYZ_sha256'])
    if len(chosen)==quota:break
   assert len(chosen)==quota;B.extend(chosen)
 assert len(A)==64 and len(B)==32 and not {r['id'] for r in A}&{r['id'] for r in B}
 assert len({byid[r['id']]['decoded_pixel_sha256'] for r in A+B})==len(A+B)==len({byid[r['id']]['canonical_XYZ_sha256'] for r in A+B})
 r16a={r['id'] for r in read(ROOT/'research/round16/input/A32_PRIVATE.json')};cases=[]
 for i,r in enumerate(A):
  c=dict(oldcases[r['id']],case_alias=f'A_{r["platform"]}_{i:03d}',subset='prior_R16_A32' if r['id'] in r16a else 'other_R15_fresh32');cases.append(c)
 for name,data in [('A64_PRIVATE.json',A),('B32_PRIVATE.json',B),('ALL96_PRIVATE.json',A+B),('A_CASES_PRIVATE.json',cases),('A_FRAME_INDEX_PRIVATE.json',{c['frame_alias']:fi[c['frame_alias']] for c in cases})]:dump(d/name,data)
 dump(d/'EXPOSURE.json',{'A64':{'prior_R16_A32':32,'other_R15_fresh32':32,'Drone':32,'Quad':32},'B32':{'Drone':16,'Quad':16,'not_R15_source96_or_regression140':True,'not_R16_A_or_B':True,'no_current_G_or_fit':True},'unique_decoded_RGB_and_canonical_XYZ_all96':True,'historical_all5912_task_scores':True,'not_blind_or_object_time_independent':True,'scene_counts':{'Drone':1,'Quad':2},'B_selection':'Fixed hash ordering of existing input-only content inventory, no caption/GT/performance filtering'})
 dump(EXP/'AUTHORIZATION.json',{'instruction':'继续修G，去掉随机错配反向，真实同类候选主体/参照物成对核验、门槛误拒审计，偏好可靠后冻结框匹配投影TTA，同信息直接重排，新观测验证，全负例保留。','no_teacher_EMA':True,'source_immutable':True,'old_R16_immutable':True})
 print('R17 registered A64+B32')
if __name__=='__main__':main()
