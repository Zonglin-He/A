import sys,pickle,collections,numpy as np,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'crossplatform/round32'))
from common32 import read,dump,rows
P=Path(__file__).resolve().parent;cache=pickle.load(open(P/'_readout_cache_PRIVATE.pkl','rb'));dif=read(P/'CPU_GPU_GATE_AGGREGATE_DIFFERENCES.json');allrows=[r for split in ['A208','DEV64','R64','S64'] for r in rows(split)];byalias={r['alias']:r for r in allrows};fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');fids={x['id_PRIVATE'] for x in fb};out=[]
for d in dif:
 r=byalias[d['alias']];A=[a for a in rows() if a['platform']==r['platform']];flag=np.array([a['id'] in fids for a in A]);near,which=cache['computed'][r['id']]['neighbors'][d['key']];order=np.argsort(-near,axis=-1,kind='stable');cpu=np.max(flag[order[:,:4]],axis=-1);candidates=[]
 for q in range(256):
  i4,i5=order[q,3:5];gap=float(near[q,i4]-near[q,i5])
  if gap<5e-7 and flag[i4]!=flag[i5]:candidates.append({'query_PRIVATE':q,'fourth_A_alias':A[i4]['alias'],'fifth_A_alias':A[i5]['alias'],'fourth_has_feedback':bool(flag[i4]),'fifth_has_feedback':bool(flag[i5]),'boundary_cosine_gap':gap,'CPU_gate':bool(cpu[q])})
 out.append(dict(d,plausible_float32_top4_boundary_candidates=candidates,actual_GPU_neighbor_indices_not_saved=True,DEV_selected_table_or_argmax_changed=False))
dump(P/'ROW_K4_NUMERICAL_BOUNDARY_DIAGNOSTICS_PRIVATE.json',out);print(out)
