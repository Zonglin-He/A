"""Final-readout interventions keeping the original continuous slots fixed."""
import numpy as np


def columns(score, unary, has_ref, native):
    independent = unary.argmax(1)
    conditioned = score.argmax(1)  # shape observation x fixed reference slot
    i = np.arange(len(native))[has_ref]
    np.testing.assert_array_equal(conditioned[i,native[i,1]],native[i,0])
    np.testing.assert_array_equal(score[i].reshape(len(i),score.shape[1]*score.shape[2]).argmax(-1),native[i,0]*score.shape[-1]+native[i,1])
    np.testing.assert_array_equal(independent[~has_ref],native[~has_ref,0])
    conditioned[~has_ref] = independent[~has_ref,None]
    return independent,conditioned
