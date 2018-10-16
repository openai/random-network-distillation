import tensorflow as tf
from baselines.common.distributions import make_pdtype
from collections import OrderedDict
from gym import spaces

def canonical_dtype(orig_dt):
    if orig_dt.kind == 'f':
        return tf.float32
    elif orig_dt.kind in 'iu':
        return tf.int32
    else:
        raise NotImplementedError

class StochasticPolicy(object):
    def __init__(self, scope, ob_space, ac_space):
        self.abs_scope = (tf.get_variable_scope().name + '/' + scope).lstrip('/')
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.pdtype = make_pdtype(ac_space)
        self.ph_new = tf.placeholder(dtype=tf.float32, shape=(None, None), name='new')
        self.ph_ob_keys = []
        self.ph_ob_dtypes = {}
        shapes = {}
        if isinstance(ob_space, spaces.Dict):
            assert isinstance(ob_space.spaces, OrderedDict)
            for key, box in ob_space.spaces.items():
                assert isinstance(box, spaces.Box)
                self.ph_ob_keys.append(key)
            # Keys must be ordered, because tf.concat(ph) depends on order. Here we don't keep OrderedDict
            # order and sort keys instead. Rationale is to give freedom to modify environment.
            self.ph_ob_keys.sort()
            for k in self.ph_ob_keys:
                self.ph_ob_dtypes[k] = ob_space.spaces[k].dtype
                shapes[k] = ob_space.spaces[k].shape
        else:
            print(ob_space)
            box = ob_space
            assert isinstance(box, spaces.Box)
            self.ph_ob_keys = [None]
            self.ph_ob_dtypes = { None: box.dtype }
            shapes = { None: box.shape }
        self.ph_ob = OrderedDict([(k, tf.placeholder(
                canonical_dtype(self.ph_ob_dtypes[k]),
                (None, None,) + tuple(shapes[k]),
                name=(('obs/%s'%k) if k is not None else 'obs')
            )) for k in self.ph_ob_keys ])
        assert list(self.ph_ob.keys())==self.ph_ob_keys, "\n%s\n%s\n" % (list(self.ph_ob.keys()), self.ph_ob_keys)
        ob_shape = tf.shape(next(iter(self.ph_ob.values())))
        self.sy_nenvs  = ob_shape[0]
        self.sy_nsteps = ob_shape[1]
        self.ph_ac = self.pdtype.sample_placeholder([None, None], name='ac')
        self.pd = self.vpred = self.ph_istate = None

    def finalize(self, pd, vpred, ph_istate=None): #pylint: disable=W0221
        self.pd = pd
        self.vpred = vpred
        self.ph_istate = ph_istate

    def ensure_observation_is_dict(self, ob):
        if self.ph_ob_keys==[None]:
            return { None: ob }
        else:
            return ob

    def call(self, ob, new, istate):
        """
        Return acs, vpred, neglogprob, nextstate
        """
        raise NotImplementedError

    def initial_state(self, n):
        raise NotImplementedError

    def update_normalization(self, ob):
        pass
