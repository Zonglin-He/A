"""Known small factor graph versus explicit joint enumeration, no data labels."""
from core87 import *
from scipy.special import softmax
N=3;A=np.eye(N);graph={'usable':True,'entities':[{'role':'T','attributes':[]},{'role':'R0','attributes':[]}],'relations':[{'subject':'T','reference':'R0','predicate':'LEFT','known_frame':True}]};ev={'means':np.zeros((3,N,13)),'counts':np.zeros((3,N)),'coverage':np.zeros((3,N)),'interval':np.array([[.1,.2],[.4,.5],[.7,.8]]),'reliability':np.ones(N),'labels':np.arange(N)};prior=np.array([.6,.3,.1]);es=np.array([[2.,1.,0.],[0.,1.,3.]])
p,info,b=target87(prior,es,graph,ev,A);u=norm87(np.sqrt(prior*softmax(es[0])));ref=np.r_[softmax(es[1])*.5,.5];gap=ev['interval'][None,:,0]-ev['interval'][:,None,1]-.01;inv=ev['interval'][:,None,0]-ev['interval'][None,:,1]-.01;K=np.ones((N,N+1));K[:,:N]=np.exp(np.tanh(np.maximum(gap,0)/.1)-np.tanh(np.maximum(inv,0)/.1))*(1-A);expected=norm87(u*(K@ref));assert np.max(abs(expected-p))<1e-12
# Exact neutral geometry has no purported LEFT/RIGHT discrimination; NULL remains.
ev['reliability']*=0;p2,i2,b2=target87(prior,es,graph,ev,A);assert 0<b2[1,-1]<1
# Query permutation equivariance; no first-member greedy/transitive grouping.
perm=np.array([2,0,1]);ep={k:(v[:,perm] if k in ['means','counts','coverage'] else v[perm] if k in ['interval','reliability','labels'] else v) for k,v in ev.items()};p3,_,_=target87(prior[perm],es[:,perm],graph,ep,A[perm][:,perm]);assert np.max(abs(p2[perm]-p3))<1e-12
dump(H87/'SYNTHETIC_AUDIT.json',{'state':'PASS_R87_PAIR_GRAPH_EXACT_ENUMERATION_NULL_AND_PERMUTATION_EQUIVARIANCE','max_small_graph_error':float(abs(expected-p).max()),'NULL_retained':True,'permutation_error':float(abs(p2[perm]-p3).max()),'GT_read':False});print('PASS_R87_SYNTHETIC')
