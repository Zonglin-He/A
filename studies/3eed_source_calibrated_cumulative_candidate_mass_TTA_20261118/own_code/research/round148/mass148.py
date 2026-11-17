"""Deterministic cumulative conditional group-mass sets; not coverage-certified."""
import numpy as np

def conditional_mass(prob,groups,scope):
 m=np.maximum(np.array([prob[g].mean() for g in groups],np.float64)[scope],1e-300);return m/m.sum()

def mass_set(mass,q):
 assert len(mass) and np.isfinite(mass).all() and 0<=q<=1
 order=np.argsort(-mass,kind='stable');cum=np.minimum(np.cumsum(mass[order]),1.);cum[-1]=1.;k=min(int(np.searchsorted(cum,q,side='left')),len(order)-1);return mass>=mass[order[k]]

def score_first_good(mass,good):
 order=np.argsort(-mass,kind='stable');cum=np.minimum(np.cumsum(mass[order]),1.);cum[-1]=1.;k=int(np.flatnonzero(good[order])[0]);return float(cum[k])
