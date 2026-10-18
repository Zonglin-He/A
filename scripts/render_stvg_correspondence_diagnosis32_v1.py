"""Publication figure from aggregate cached diagnostics; no media or GT geometry."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_stvg_correspondence32_v1 import PUB,read
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
z=read(PUB/'RESULTS.json');fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for col,ds in enumerate(['hc2','vidstg']):
 g=z['groups'][ds]['results']['unobserved'];arms=['real','near_shift','far_shift','interpolation'];x=np.arange(len(arms))
 for j,(met,label)in enumerate([('rank_delta','Original strength'),('matched_strength_rank_delta','Matched strength')]):axs[0,col].bar(x+(j-.5)*.35,[100*g[a][met]['mean']for a in arms],width=.35,label=label)
 axs[0,col].axhline(0,color='black',lw=.7);axs[0,col].set(xticks=x,xticklabels=['Real','Near shift','Far shift','Geometry'],ylabel='Candidate IoU rank increment (pp)',title=ds+' same candidates, n=16 parents');axs[0,col].legend()
 arms=['interpolation','interpolation_raster','real','half_grid_correspondence'];x=np.arange(len(arms));vals=[g[a]['rank_delta']for a in arms];means=np.array([v['mean']for v in vals])*100;ci=np.array([v['CI95']for v in vals])*100;axs[1,col].bar(x,means,color=['#2a9d8f','#8ab17d','#e9c46a','#f4a261']);axs[1,col].errorbar(x,means,yerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt='none',ecolor='black',capsize=3);axs[1,col].axhline(0,color='black',lw=.7);axs[1,col].set(xticks=x,xticklabels=['Continuous\ngeometry','Rasterized\nsame geometry','Real\nnative grid','Real\nhalf grid'],ylabel='Original-strength rank increment (pp)',title=ds+' scoring representation and coarsening')
fig.suptitle('Cached failure decomposition: strength, temporal control and grid representation')
fig.savefig(PUB/'DIAGNOSIS.svg');fig.savefig(PUB/'DIAGNOSIS.png',dpi=140)
