"""Minimal candidate G: one image-text evidence value per RGB instance.
Independent mask/camera association is supplied by the caller. Nothing here
estimates boxes, parses references, calibrates with gray images, centers a
residual on the source winner, or updates parameters.
"""
import numpy as np
def simple_fusion(source_scores,query_to_instance_weights,instance_image_text_scores,weight):
 source=np.asarray(source_scores);w=np.asarray(query_to_instance_weights);e=np.asarray(instance_image_text_scores)
 if source.ndim!=1 or w.shape!=(len(source),len(e)) or e.ndim!=1:raise ValueError('Expected source[Q], weights[Q,M], evidence[M]')
 if not np.isfinite(source).all() or not np.isfinite(w).all() or not np.isfinite(e).all() or not np.isfinite(weight) or weight<0:raise ValueError('Need finite scores, nonnegative scalar weight')
 mass=w.sum(1)
 if (w<0).any() or not np.all((abs(mass)<1e-8)|(abs(mass-1)<1e-6)):raise ValueError('Association rows must be normalized or unmapped zero')
 scores=source+float(weight)*(w@e)
 return scores,int(scores.argmax())
