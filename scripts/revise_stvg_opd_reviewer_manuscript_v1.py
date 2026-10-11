"""Create an additive reviewer-response working draft; preserve supplied source.

Usage: python script.py input-source output-tex. This script edits prose and
displays already closed results. It never runs a model or accesses GT geometry.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT/'results/stvg_opd_mechanism_review/2026-10-11'
BASE = ROOT/'artifacts/stvg_opd_mechanism_review_v1'


def run(source, destination):
    source, destination = Path(source), Path(destination)
    original = source.read_bytes()
    text = original.decode('utf-8').replace('\r\n', '\n')
    changes = []

    def replace(old, new, claim):
        nonlocal text
        assert text.count(old) == 1, (claim, text.count(old))
        text = text.replace(old, new)
        changes.append(claim)

    def section(start, end, new, claim):
        nonlocal text
        assert text.count(start) == text.count(end) == 1
        a, b = text.index(start), text.index(end)
        assert b > a
        text = text[:a] + new + '\n\n' + text[b:]
        changes.append(claim)

    replace(r'\maketitle'+'\n\n'+r'\section{Introduction}', r'''\maketitle

\begin{abstract}
Spatio-temporal video grounding must recover both an event interval and the
trajectory of a language-specified instance under deployment shifts.
We study spatial test-time adaptation using sparse observations from one
frozen grounding specialist, while retaining the student's native event
interval. Our framework samples localization actions around the current
student prediction, reweights them by their overlap with the specialist box,
and fits the resulting target through a 1,792-parameter decoder interface.
A query residual resets after each arrival, whereas a fraction of the
LayerNorm displacement initializes subsequent queries.
On the full official cross-domain populations, the system increases mean
video IoU from 20.017 to 22.583 on HC-STVG-v2 and from 11.129 to 14.097 on
VidSTG. Matched mechanism comparisons show a conditional tradeoff:
local IoU feedback limits regression toward low-overlap references in some
settings, but can impede correction when the specialist is accurate and
sampled actions receive zero reward. The framework provides a compact
spatial adaptation interface; direct regression, state-reset comparisons,
and offline temporal oracles delimit the contribution of each design.
\end{abstract}

\section{Introduction}''', 'add evidence-bounded Abstract')

    replace('a frozen specialist evaluates student-generated box\ncandidates within the native interval, and the student\'s updated decoder',
            'a frozen specialist supplies reference boxes within the native\ninterval, geometric overlap evaluates student-generated candidates, and the student\'s updated decoder',
            'separate frozen box generation from geometric action evaluation')
    replace('are sampled around the current student prediction and evaluated by the\ngrounding specialist.',
            'are sampled around the current student prediction and evaluated by\ngeometric agreement with the fixed specialist reference.',
            'avoid independent semantic teacher evaluation claim')
    replace('current policy, while Grounding DINO evaluates these actions and induces a\nfeedback-improved policy.',
            'current policy, while IoU with the fixed Grounding DINO box evaluates\nthese actions and induces a feedback-improved policy.',
            'state actual fixed-reference reward interface')
    replace('Given the inherited model state, the student first predicts\n$Y_n^0=(I_n^0,B_n^0)$ with native interval\n$I_n^0=[s_n^0,e_n^0]$.',
            'For each arrival, the original native tube provides the fixed interval\n$I_n^0=[s_n^0,e_n^0)$. The inherited adaptive state provides the\nspatial Before prediction; it does not redefine this cached temporal scope.\nSpatial updates retain $I_n^0$ in the final output, so an event portion\noutside this interval cannot be recovered by this method.',
            'distinguish cached Native WHEN from inherited Before spatial state')
    replace('We collect these observations as\n$\\Omega_n=\\{(t_{n,k},\\bar b_{n,k})\\}$.',
            r'''We collect these observations as
$\Omega_n=\{(t_{n,k},\bar b_{n,k})\}$.
The fixed admission rule requires mixed entity/phrase confidence at least
0.35 and a margin of at least 0.05 over the next distinct candidate after
duplicate suppression at IoU 0.5. These rules use no target GT.
When no observation is admitted, no spatial optimizer step is taken and
After equals the inherited Before prediction. Before can differ from the
frozen source prediction because preceding arrivals may have changed
LayerNorm state; the query residual and optimizer still reset.''',
            'make exact admission and no-reference behavior explicit')
    section('Sparse specialist observations alone do not provide a complete video\ntrajectory.',
            r'\paragraph{Lightweight spatial adaptation interface.}',
            r'''Sparse specialist observations supply reference boxes rather than a
complete trajectory. We compare local student-generated actions by their
IoU with each fixed reference, then fit the resulting weighted target.
This interface differs from directly minimizing L1/GIoU to the reference
in its local sampling and iterative reward reweighting. Its value relative
to direct regression is an empirical question evaluated with a matched
Direct control; distributional supervision alone does not establish a
stronger or semantically richer teacher signal.''',
            'remove assumed superiority of distributional over point supervision')
    replace('Having generated student actions, we next ask which of them better explain the\nspecialist observation. For each sampled action, Grounding DINO provides the\nspatial reward',
            'Having generated student actions, we compare them with the same fixed\nspecialist observation. For each sampled action, our geometric reward is',
            'correct reward producer description')
    replace('Crucially, the distillation target evolves with the student.',
            r'''For fixed action samples and detached weights, the sample-weighted
Gaussian likelihood has the equivalent weighted-regression form
\begin{equation}
\mathcal L_{\mathrm{sample}}^{(s)}(\phi)
=\frac{1}{2\sigma^2|\Omega_n|}
\sum_{t\in\Omega_n}\left\|\mu_{n,t}(\phi)
-\sum_m w_{n,t,m}^{(s)}\operatorname{sg}[z_{n,t,m}^{(s)}]\right\|_2^2+C.
\label{eq:weighted-regression}
\end{equation}
Here $C$ is independent of the updated mean. The distinction from Direct
therefore lies in the locally constructed target and its resampling across
rounds, rather than escaping regression mathematically. If all sampled IoU
rewards are zero at one observation, its weights are uniform and its
antithetic centered correction vanishes at the sampling mean. This does
not imply a query-wide no-op: other observations and shared decoder state
can still move its box. Zero reward can limit both correction toward an
accurate distant reference and movement toward a poor reference.

The distillation target evolves with the student.''',
            'show fixed-weight regression equivalence and finite-action zero reward boundary')
    replace('The settings are fixed using development subsets of 32 parent\nvideos per target; no evaluation labels are used during an\nadaptation stream.',
            'The settings are fixed using development subsets of 32 parent\nvideos per target; no evaluation labels are used during an\nadaptation stream. We separately report analyses excluding these\nparents. The remaining sources have historical evaluation exposure\nand are not described as a newly untouched confirmation population.',
            'retain development selection and historical exposure boundary')
    replace('STVG adaptations of Tent~\\cite{tent} and SAR~\\cite{sar}.',
            'STVG adaptations of Tent~\\cite{tent} and SAR~\\cite{sar},\nand parameter-free DINO Refine using exactly the same four\nadmitted references. DINO Refine interpolates absolute center--size\nboxes between admitted anchors with nearest endpoint extension;\nit is not an update at only four frame positions.',
            'restore exact DINO Refine comparator definition')
    replace('All values are percentages. The in-domain supervised reference',
            'All values are percentages. Tent/SAR here retain the original\n        evaluated configurations; separately calibrated configurations are\n        not substituted before their complete formal evaluation.\n        The in-domain supervised reference',
            'label existing baseline configuration honestly while calibrated job is pending')
    replace('SAR-STVG            & 19.939 & 40.152 & 48.556 & 27.465 & 7.486 \\\\',
            'SAR-STVG            & 19.939 & 40.152 & 48.556 & 27.465 & 7.486 \\\\\n\t\tDINO Refine          & 20.673 & 40.311 & 43.631 & 28.920 & 9.075 \\\\',
            'restore full-query HC DINO row')
    replace('SAR-STVG            & 11.129 & 31.703 & 32.758 & 12.831 & 3.300 \\\\',
            'SAR-STVG            & 11.129 & 31.703 & 32.758 & 12.831 & 3.300 \\\\\n\t\tDINO Refine          & 13.985 & 31.703 & 35.710 & \\textbf{18.024} & 6.920 \\\\',
            'restore full-query Vid DINO row including better R@0.3')
    replace(r'\textbf{17.904} & \textbf{7.037}',
            r'17.904 & \textbf{7.037}', 'correct Vid R@0.3 best-value formatting')
    replace('We therefore retain the fixed\nfour-frame configuration for the main experiments.',
            'The main four-frame configuration was fixed before this\nbudget evaluation; these results do not select a new test-set budget.',
            'avoid implying P5 formal GT chose K=4')

    a=text.index(r'\label{tab:opd-ablation}'); b=text.index(r'\end{table*}', a)
    old=text[a:b]
    new=old.replace(r'\setlength{\tabcolsep}{11pt}',r'\setlength{\tabcolsep}{5pt}')
    new=new.replace(r'\begin{tabular}{lcc}',r'\begin{tabular}{lccc}')
    new=new.replace('Setting & Full $-$ Shuffled & Full $-$ Fixed Rollout',
                    'Setting & Full $-$ Direct & Full $-$ Shuffled & Full $-$ Fixed Rollout')
    for oldrow, newrow in [
        ('$+3.837\\;[2.257,5.335]$', '$+0.576\\;[-0.236,1.433]$ & $+3.837\\;[2.257,5.335]$'),
        ('$+8.820\\;[6.861,10.979]$', '$-0.213\\;[-0.625,0.216]$ & $+8.820\\;[6.861,10.979]$'),
        ('$+6.320\\;[4.595,7.991]$', '$+0.603\\;[-1.069,2.267]$ & $+6.320\\;[4.595,7.991]$'),
        ('$+10.222\\;[7.455,13.033]$', '$+0.523\\;[-0.343,1.494]$ & $+10.222\\;[7.455,13.033]$')]:
        assert new.count(oldrow)==1
        new=new.replace(oldrow,newrow)
    replace(old,new,'restore matched P2 Direct contrast without mixing full-query populations')
    replace('these choices with two controlled variants.',
            'these choices with three controlled variants. \\emph{Direct}\nminimizes L1/GIoU (weights 5/2) to the same admitted references,\nusing the same trainable interface, rounds and stream lifecycle.',
            'define fair Direct L1/GIoU control')
    replace(r'\paragraph{Feedback correspondence.}',r'''\paragraph{Direct regression.}
The Full-minus-Direct intervals in Table~\ref{tab:opd-ablation}
span zero in all four settings. The paired means are $+0.576$
points on cross-domain HC-STVG-v2 and $-0.213$ on cross-domain
VidSTG. These comparisons do not establish stable average OPD
superiority; reward correspondence and policy-refresh ablations test
specific choices within OPD rather than replace this direct comparator.

\paragraph{Feedback correspondence.}''',
            'state actual direct-regression evidence boundary')

    section(r'\paragraph{Corrections beyond observed frames.}',
            r'\paragraph{Qualitative behavior.}',r'''\paragraph{Corrections beyond observed frames.}
We separate admitted observed frames, unobserved real decoder frames
inside Native WHEN, interpolation-only frames inside Native, and
annotated event frames outside Native. On the matched P2 population
excluding development parents, the current After-minus-Before change
on unobserved real decoder frames is $+5.419$ points on cross-domain
VidSTG (95\% parent CI $[2.594,8.278]$) and $-0.581$ on
cross-domain HC-STVG-v2 ($[-3.238,1.865]$). The corresponding
After-minus-DINO differences are $+2.647$ ($[-0.146,5.777]$)
and $+4.394$ ($[0.193,8.450]$). These contrasts distinguish the
current spatial update from the complete online model's advantage
over parameter-free refinement. Common observed/unobserved parent
intersections are reported separately: for HC, the latter interval
then spans zero. Interpolation alone is not evidence of a learned
decoder change, and geometric changes outside Native WHEN do not
recover temporal coverage.

\paragraph{Expert quality and local actionability.}
Using the actual saved rewards, we stratify expert frames by
expert--GT IoU $Q\geq0.5$ or $Q<0.5$ and by whether any first-round
action has positive reward. In the low-$Q$/all-zero strata,
Full-minus-Direct current frame-IoU changes are $+11.520$ points
on cross-domain HC ($[4.066,19.682]$, 22 parents) and $+9.751$
under HC corruption ($[1.474,18.744]$, 32 parents). The analogous
Vid intervals span zero. However, high-$Q$/all-zero frames favor
Direct in the HC comparisons, by $29.329$ and $40.875$ points on
only 5 and 11 parents. These descriptive geometric strata suggest
a conditional conservative-update tradeoff, not a universal
reliability advantage or verified identity-error detector.
Full and Direct have their own inherited Before states.
All-zero first-round reward occurs in 1.274--6.593\% of the
high-$Q$ observations across the four settings. It remains a
correction failure mode, even though the predeclared resource
trigger for a changed-reward pilot is not met.''',
            'replace mixed observed/other groups with actual decoder, interpolation and risk evidence')
    section(r'\paragraph{Qualitative behavior.}',
            r'\paragraph{Complementary temporal corrections.}',r'''\paragraph{Qualitative behavior.}
Aligned Native/Before/After/GT examples show both improvements and
regressions, with the four real specialist observations and sampled
intermediate frames marked. Specialist agreement can increase while
GT overlap decreases. Geometric disagreement alone does not establish
an identity switch; such an attribution requires instance evidence
and inspection of the actual sequence.''',
            'remove unsupported identity failure prevalence claim')
    replace(r'\paragraph{Complementary temporal corrections.}',r'''\paragraph{Offline spatial and temporal headroom.}
On the matched 128-parent cross-domain populations, the Native,
GT-space/Native-time, and Native-space/GT-time video IoUs are
22.156, 42.239, and 51.417\% for HC, and 14.466, 36.614,
and 38.213\% for VidSTG. The temporal oracle uses only the
original native trajectory and its supported interpolation;
unpredicted frames are never filled with GT spatial boxes.
Temporal headroom is at least as large as spatial headroom in
these settings, so the motivating spatial errors do not imply a
universal dominant spatial bottleneck. Native WHEN covers about
56\% of annotated event support. These offline interventions are
diagnostic ceilings, not deployable comparisons or an additive
decomposition of task error.

\paragraph{Complementary temporal corrections.}''',
            'report offline temporal limitation without GT-filled missing native boxes')
    replace('within the native event scope, while jointly adapting\nboth components remains a distinct research direction.',
            'within the native event scope. Joint temporal adaptation is not\npart of the evaluated primary method.',
            'avoid adding an untested architecture as a claimed result')
    text = '% Reviewer-response working copy; figures/style/bibliography require the original project.\n' + text
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    destination.write_text(text)
    assert source.read_bytes()==original
    assert text.count(r'\begin{abstract}')==1 and text.count('DINO Refine')>=3
    assert 'Full $-$ Direct' in text and r'\textbf{17.904}' not in text
    assert text.count(r'\begin{')==text.count(r'\end{')
    result=dict(status='editable_working_source_written',
                input_sha256=hashlib.sha256(original).hexdigest(),
                output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                changes=changes, source_preserved=True, primary_method_changed=False,
                new_model_calls=0, calibrated_baselines_pending=True,
                cross_parent_LN_actual_GPU_pending=True,
                end_to_end_uncached_latency_pending=True,
                page_limit_or_final_submission_layout_not_verified=True)
    (BASE/'MANUSCRIPT_REVISION_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='changes'}))


if __name__=='__main__':
    run(*sys.argv[1:])
