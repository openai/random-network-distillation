from rl_algs.ppo_metal.policies.cnn_gru_policy import CnnGruPolicy
from rl_algs.ppo_metal.policies.cnn_policy import CnnPolicy
from rl_algs.ppo_metal.policies.resnet_policy import ResnetPolicy, ResnetGruPolicy
import functools

def policy_gen(scope_name, ob_space, ac_space, hps):
    arch = hps.pop('arch')

    if arch == 'cnn':
        args = { k: hps.pop(k) for k in [] if k in hps }
        stochpol_fn = functools.partial(
            CnnPolicy, scope=scope_name, ob_space=ob_space, ac_space=ac_space, policy_size=hps.pop('policy_size'), **args)

    elif arch == 'cnngru':
        args = { k: hps.pop(k) for k in ['hidsize', 'memsize', 'extrahid', 'rec_gate_init', 'maxpool'] if k in hps }
        stochpol_fn = functools.partial(
            CnnGruPolicy, scope=scope_name, ob_space=ob_space, ac_space=ac_space, policy_size=hps.pop('policy_size'), **args)

    elif arch == 'resnet':
        args = { k: hps.pop(k) for k in ['resnet_flm', 'resnet_num_blocks', 'resnet_output_units'] if k in hps }
        stochpol_fn = functools.partial(
            ResnetPolicy, scope=scope_name, ob_space=ob_space, ac_space=ac_space, **args)

    elif arch == 'resnetgru':
        args = { k: hps.pop(k) for k in ['hidsize', 'memsize', 'extrahid', 'rec_gate_init', 'resnet_flm', 'resnet_num_blocks', 'resnet_output_units'] if k in hps }
        stochpol_fn = functools.partial(
            ResnetGruPolicy, scope=scope_name, ob_space=ob_space, ac_space=ac_space, **args)

    else:
        assert 0, "Unknown policy architecture '%s'" % arch

    return stochpol_fn
