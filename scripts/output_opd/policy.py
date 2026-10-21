"""Three-update output-distribution OPD; candidates share one student graph.

The energy teacher is q_E(X) proportional to q0(X) exp(U(B(X))/tau).
OPD estimates its reverse-KL gradient with conditional score functions and
factor returns. DirectMC estimates the identical objective pathwise.
"""
from __future__ import annotations
import copy,gc,hashlib,math,time
import torch
from scripts.output_opd.replay import FrozenTrajectoryReplay
from scripts.output_opd.distribution import JointTrajectoryGaussian,antithetic_noise,independent_pair_baseline
from scripts.output_opd.feedback import TrackedFeedback
from scripts.trajectory_fast.policy import EventRecorder

ARMS=('OutputOPD','Direct','DirectMC')


def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v)for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [cpu(v)for v in x]
    return x


def brief(u):
    out={k:float(u[k].detach())for k in ('U','mask','link','guard','guard_mean','guard_max','obs','unobs','d_mean','d_max')}
    if not all(math.isfinite(v)for v in out.values()):raise FloatingPointError('Nonfinite actual trajectory utility')
    return out


def fit(model,cache,evidence,key,arm,cfg):
    assert arm in ARMS and cfg['rounds']==3 and cfg['candidates']==4
    device=next(model.ground_decoder.decoder.parameters()).device
    assert device.type=='cuda'
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    timing=EventRecorder(device)
    with timing.record('replay_construction_no_forward'):
        replay=FrozenTrajectoryReplay(model,cache,evidence.W,full_reference=True)
        law=JointTrajectoryGaussian(replay.E,cache['frame_ids'],[cache['frame_ids'][p]for p in cache['planned_positions']],sigma=cfg['sigma'])
        feedback=TrackedFeedback(evidence)
    z=torch.nn.Parameter(torch.zeros(8,256,device=device))
    opt=torch.optim.Adam([z],lr=cfg['lr'],betas=(.9,.999),eps=1e-8)
    path=[];rounds=[];final=None;backwards=accepted=rejected=0
    try:
        for r in range(3):
            before=z.detach().clone();before_opt=copy.deepcopy(opt.state_dict())
            opt.zero_grad(set_to_none=True)
            with timing.record('student_forward'):
                boxes=replay.boxes(z)
            with timing.record('actual_current_reward'):
                actual=evidence.utility(boxes,replay.E)
            current=brief(actual)
            if r==0:assert torch.equal(boxes.detach(),replay.E)
            if final is not None:assert torch.equal(boxes.detach(),final),'Validation/training forward changed'
            final=boxes.detach();rr=dict(round=r,before=current,state_before=cpu(before))
            with timing.record('joint_output_distribution'):
                mu=law.chart(boxes);kl=law.kl_reference(mu)
            if arm=='Direct':
                loss=kl-actual['U']/cfg['tau']
                rr.update(KL_reference=float(kl.detach()),training_U=current['U'])
            else:
                eps=antithetic_noise(key,r,len(cache['frame_ids']),device=device,dtype=mu.dtype)
                with timing.record('joint_output_distribution'):
                    X=law.sample(mu,eps,detach_mean=arm=='OutputOPD')
                    candidates=law.inverse(X)
                if arm=='OutputOPD':
                    with torch.no_grad(),timing.record('candidate_feedback',evaluations=4):
                        teacher=feedback.evaluate(candidates,replay.E)
                    returns=teacher['reward_to_go'].detach()
                    advantage=returns-independent_pair_baseline(returns)
                    with timing.record('distillation_loss'):
                        logp=law.log_prob_terms(X.detach(),mu)
                        loss=kl-(advantage.detach()*logp).sum(-1).mean()/cfg['tau']
                else:
                    with timing.record('candidate_feedback',evaluations=4):
                        teacher=feedback.evaluate(candidates,replay.E)
                    loss=kl-teacher['U'].mean()/cfg['tau']
                rr.update(candidate_X=cpu(X),candidate_boxes=cpu(candidates),noise=cpu(eps),teacher=cpu(teacher),KL_reference=float(kl.detach()),candidate_feedback_uses_one_student_forward=True)
                if arm=='OutputOPD':rr.update(advantage=cpu(advantage),conditional_log_prob=cpu(logp))
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite adaptation loss')
            with timing.record('complete_backward'):
                loss.backward();backwards+=1
            if z.grad is None or not bool(torch.isfinite(z.grad).all()):raise FloatingPointError('Nonfinite content gradient')
            grad_norm=float(z.grad.norm());rr.update(loss=float(loss.detach()),gradient_norm=grad_norm)
            with timing.record('temporary_state_optimizer'):
                opt.step()
            rr['proposal_state']=cpu(z)
            valid=True
            with torch.no_grad():
                try:
                    with timing.record('updated_student_forward'):
                        proposal=replay.boxes(z)
                    with timing.record('actual_updated_reward'):
                        proposal_u=evidence.utility(proposal,replay.E)
                    proposed=brief(proposal_u)
                except FloatingPointError:
                    valid=False;proposed=None
            found=valid and proposed['U']>current['U']+cfg['accept_epsilon']
            if found:
                accepted+=1;final=proposal.detach();after=proposed
            else:
                rejected+=1
                with torch.no_grad():z.copy_(before)
                opt.load_state_dict(before_opt);after=current
            rr.update(accepted=found,proposal=proposed,proposal_numerically_valid=valid,after=after,state_after=cpu(z),full_trajectory_evals=replay.evals)
            rounds.append(rr);path.append(dict(round=r+1,state=cpu(z),boxes=cpu(final),**after))
            del loss,boxes,mu,kl,actual
            if arm!='Direct':del X,candidates,eps,teacher
            if not found:break
        assert replay.evals==2*len(rounds) and replay.evals<=6 and backwards<=3
        assert replay.setup_evals==0
        replay.assert_frozen();timing.end_arm();torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
        profile=timing.summary_after_synchronize(seconds)
        return dict(arm=arm,final=cpu(final),final_z=cpu(z),final_eta=cpu(replay.expand(z)),path=path,rounds=rounds,initial_checks=replay.initial_checks,
                    output_distribution=dict(sigma=cfg['sigma'],tau=cfg['tau'],correlation_rule='physical AR1; decay=median observed physical gap',reference='fixed ERef chart zero mean',trainable_covariance=False,probability_space='unclipped normalized center and log-scale chart',teacher='q0 exp(U/tau) normalized implicitly'),
                    evidence=evidence.metadata,profiling=profile,
                    costs=dict(seconds=seconds,full_trajectory_evals=replay.evals,native_offset_decoder_calls=2*replay.evals,setup_full_trajectory_evals=0,setup_native_offset_decoder_calls=0,decoder_backwards=backwards,state_only_backwards=0,updates=len(rounds),accepted=accepted,rejected=rejected,accepted_updates=accepted,output_candidates=4*len(rounds)if arm!='Direct'else 0,allowance=6,peak_GPU_bytes=torch.cuda.max_memory_allocated(),stop_reason='rejected_and_restored'if rejected else'three_updates_completed'))
    finally:
        opt.zero_grad(set_to_none=True);del replay,opt;gc.collect();torch.cuda.empty_cache()
