"""Globally maximize upright 3D IoU over axis-aligned horizontal center pieces."""
import numpy as np

from center_sets import axes, gaps
from score_ablation import many_iou


SIGNS = np.array(((-1.,-1.), (-1.,1.), (1.,-1.), (1.,1.)))
CHECK_TOL = 1.e-8


def center_pieces(bounds, exclusions):
    """Closed rectangles whose union equals the closed domain minus open exclusions."""
    if bounds is None:
        return []
    lower, upper = np.asarray(bounds, float)
    if not exclusions:
        return [(lower, upper)]
    forb = np.asarray(exclusions, float)
    start, end = forb[:,0], forb[:,1]
    knots = np.unique(np.r_[lower[0], upper[0], np.clip(start[:,0],lower[0],upper[0]),
                            np.clip(end[:,0],lower[0],upper[0])])
    pieces = []
    for l, h in list(zip(knots[:-1], knots[1:])) + [(x,x) for x in knots]:
        probe = l+(h-l)/2
        active = (start[:,0] < probe) & (probe < end[:,0])
        for a,b in gaps(lower[1], upper[1], start[active,1], end[active,1]):
            pieces.append((np.array((l,a)), np.array((h,b))))
    return pieces


def events(offset, velocity):
    ratios = np.divide(offset, velocity, out=np.zeros_like(offset), where=velocity!=0)
    return ratios[(ratios > 0.) & (ratios < 1.)]


class HorizontalSweep:
    """One frozen candidate pose, with horizontal translation in its own local axes."""
    def __init__(self, box, truth):
        self.box = np.array(box, float)
        self.basis = axes(box[6])[:2,:2]
        self.local_box = np.r_[np.zeros(3),box[3:6],0.]
        self.truth = np.array(truth, float)
        self.truth[:2] = (truth[:2]-box[:2]) @ self.basis
        self.truth[2] -= box[2]
        self.truth[6] -= box[6]
        self.gt_axes = axes(self.truth[6])[:2,:2]
        self.center = self.truth[:2]
        self.offsets = SIGNS * box[3:5]/2
        self.gt_vertices = (SIGNS * truth[3:5]/2) @ self.gt_axes.T + self.center
        signs = np.array((-1.,1.))[:,None]
        self.box_sides = signs * box[3:5]/2
        self.gt_sides = signs * truth[3:5]/2
        self.free_value = float(self.values([self.center])[0])

    def values(self, xy):
        moves = np.repeat(self.local_box[None], len(xy), axis=0)
        moves[:,:2] = xy
        return many_iou(moves, self.truth)

    def translate(self, xy):
        new = self.box.copy()
        new[:2] += xy @ self.basis.T
        return new

    def segment_knots(self, start, end):
        direction = end-start
        moving_in_truth = (self.offsets+start-self.center) @ self.gt_axes
        cut_moving = events(self.gt_sides[None,:,:]-moving_in_truth[:,None,:], direction @ self.gt_axes)
        cut_fixed = events(self.gt_vertices[:,None,:]-start-self.box_sides[None,:,:], direction)
        knots = np.unique(np.r_[0., cut_moving, cut_fixed, 1.])
        return start + knots[:,None]*direction

    def optimum(self, pieces):
        """Return a maximizing center and IoU, checking the quadratic model on every edge segment."""
        if not pieces:
            return None, None, 0.
        if self.free_value == 0.:
            return pieces[0][0], 0., 0.
        if any(((lower <= self.center) & (self.center <= upper)).all() for lower,upper in pieces):
            return self.center, self.free_value, 0.
        edges = set()
        for lo,hi in pieces:
            corners = [tuple(lo), (hi[0],lo[1]), tuple(hi), (lo[0],hi[1])]
            for p,q in zip(corners, corners[1:]+corners[:1]):
                edges.add(tuple(sorted((p,q))))
        segments = [self.segment_knots(np.array(l),np.array(r)) for l,r in sorted(edges)]
        lb = np.concatenate([s[:-1] for s in segments])
        ub = np.concatenate([s[1:] for s in segments])
        delta = ub-lb
        xy = np.concatenate([lb+f*delta for f in (0., .25, .5, .75, 1.)])
        scores = self.values(xy)
        # Fixed volumes make intersection divided by volume sum the quadratic quantity.
        y0,yq0,ym,yq1,y1 = np.split(scores/(1.+scores), 5)
        a = 2.*(y1+y0-2.*ym)
        b = y1-y0-a
        error = max(float(np.max(np.abs(a*.25**2+b*.25+y0-yq0))),
                    float(np.max(np.abs(a*.75**2+b*.75+y0-yq1))))
        if error > CHECK_TOL:
            raise ValueError(f"Piecewise-quadratic intersection check failed: {error:g}")
        root = np.divide(-b,2.*a,out=np.zeros_like(a),where=a<0.)
        between = (root>0.) & (root<1.)
        vertices = lb[between] + root[between,None]*delta[between]
        vertices_scores = self.values(vertices)
        best_endpoint = int(np.argmax(scores))
        score = float(scores[best_endpoint])
        if len(vertices_scores) and float(vertices_scores.max()) > score:
            best = int(np.argmax(vertices_scores))
            return vertices[best], float(vertices_scores[best]), error
        return xy[best_endpoint], score, error
