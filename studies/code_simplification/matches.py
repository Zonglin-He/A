"""One query-local matching service; unchanged cosine/top20/softmax rule.

The inactive research adapter is verified against every original affinity and
geometry tensor before any production adoption. No propagation component is
removed. Cache lifetime ends after one geometry construction.
"""
import torch


class PairMatches:
    def __init__(self):
        self.similarities = {}
        self.weights = {}
        self.inputs = None
        self.checked_inputs = set()
        self.calls = self.dot_products = self.hits = 0

    def __call__(self, normalized, valid, target, source):
        identity = (normalized.data_ptr(), valid.data_ptr())
        if identity not in self.checked_inputs:
            if self.inputs is None:
                self.inputs = (normalized, valid)
            else:
                assert torch.equal(normalized, self.inputs[0])
                assert torch.equal(valid, self.inputs[1])
            self.checked_inputs.add(identity)
        self.calls += 1
        pair = (int(target), int(source))
        if pair in self.weights:
            self.hits += 1
            return self.weights[pair]

        ti = torch.nonzero(valid[target], as_tuple=False).flatten()
        si = torch.nonzero(valid[source], as_tuple=False).flatten()
        result = torch.zeros(normalized.shape[1], normalized.shape[1], dtype=torch.float64)
        if len(ti) and len(si):
            if pair in self.similarities:
                cosine = self.similarities[pair]
            elif pair[::-1] in self.similarities:
                cosine = self.similarities[pair[::-1]].T
            else:
                cosine = normalized[target, ti] @ normalized[source, si].T
                self.similarities[pair] = cosine
                self.dot_products += 1
            scores, indices = torch.topk(cosine, min(20, len(si)), dim=1, sorted=False)
            probability = (scores / .1).softmax(1).double()
            probability /= probability.sum(1, keepdim=True)
            result[ti[:, None].expand_as(indices), si[indices]] = probability
        self.weights[pair] = result
        return result

    def stats(self):
        arrays = [*self.similarities.values(), *self.weights.values()]
        return dict(requests=self.calls, dot_products=self.dot_products, exact_pair_hits=self.hits,
                    retained_cache_bytes=sum(x.numel()*x.element_size() for x in arrays))
