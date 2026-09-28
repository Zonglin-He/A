"""Precommitted diagnosis screens; associations are not causal proof."""
def quadrant(oracle_delta,learned_delta,eps=1e-10):
    def sign(x):return 'gain' if x>eps else 'harm' if x < -eps else 'flat'
    return sign(oracle_delta)+'/'+sign(learned_delta)

def choose_branch(oracle,seeds,rescue=None):
    advantage=(oracle['v_mean']>0 and oracle['v_lower']>0 and oracle['t_mean']>=0 and oracle['s_mean']>=0)
    mapping=advantage and all(s['oracle_minus_learned_lower']>0 and (
        (s['cos_mean'] is not None and s['cos_median'] is not None and s['cos_mean']<=.1 and s['cos_median']<=.1)
        or any(s['oracle_descent'][b]-s['learned_descent'][b]>=.20 for b in ('event','spatial')
               if s['oracle_descent'][b] is not None and s['learned_descent'][b] is not None)) for s in seeds)
    trust=advantage and all(s['cos_mean'] is not None and s['cos_median'] is not None and s['cos_mean']>=.30 and s['cos_median']>=.30
        and all(s['learned_descent'][b] is not None and s['learned_descent'][b]>=.75 for b in ('event','spatial'))
        and s['saturated_fraction']>=.90 and s['v_mean']<=0 for s in seeds)
    if mapping and trust:return dict(DECISION='INCONCLUSIVE',recommendation=None,reason='Both screens met; no arbitrary priority or favorable seed')
    if mapping:return dict(DECISION='A',recommendation='A',reason='Oracle native advantage, poor learned mapping; conditioning explanation remains a hypothesis')
    if trust:return dict(DECISION='B',recommendation='B',reason='Aligned local direction with saturated finite failure; magnitude/no-op mechanism needs intervention')
    one_step_failed=oracle['v_mean']<=0 or oracle['v_lower']<=0
    if one_step_failed and rescue is not None:
        if rescue.get('status')!='independently_audited':raise ValueError('Unmeasured rescue cannot receive scientific decision')
        if rescue['union_qualified']:return dict(DECISION='C1',recommendation=None,reason='Measured iterative union passes locked native gate')
        if rescue['free_qualified']:return dict(DECISION='C2',recommendation=None,reason='Measured union fails, free merger passes locked native gate')
        return dict(DECISION='C3',recommendation=None,reason='Both measured rescue configurations fail; only tested interface implicated')
    return dict(DECISION='INCONCLUSIVE',recommendation='C' if one_step_failed else None,
        reason='Rescue not run; C is the single proposed next test, not a C1/C2/C3 result' if one_step_failed else 'Mixed evidence does not identify a single mechanism')
