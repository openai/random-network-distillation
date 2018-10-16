import numpy as np
import tensorflow as tf
from baselines import logger
from utils import fc, conv, ortho_init
from stochastic_policy import StochasticPolicy
from tf_util import get_available_gpus
from mpi_util import RunningMeanStd


def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]: size *= shapel.value
    return tf.reshape(x, (-1, size))

def _fcnobias(x, scope, nh, *, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        return tf.matmul(x, w)
def _normalize(x):
    eps = 1e-5
    mean, var = tf.nn.moments(x, axes=(-1,), keepdims=True)
    return (x - mean) / tf.sqrt(var + eps)


class CnnPolicy(StochasticPolicy):
    def __init__(self, scope, ob_space, ac_space,
                 policy_size='normal', maxpool=False, extrahid=True, hidsize=128, memsize=128, rec_gate_init=0.0,
                 update_ob_stats_independently_per_gpu=True,
                 proportion_of_exp_used_for_predictor_update=1.,
                 dynamics_bonus = False,
                 ):
        StochasticPolicy.__init__(self, scope, ob_space, ac_space)
        self.proportion_of_exp_used_for_predictor_update = proportion_of_exp_used_for_predictor_update
        enlargement = {
            'small': 1,
            'normal': 2,
            'large': 4
        }[policy_size]
        rep_size = 512
        self.ph_mean = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obmean")
        self.ph_std = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obstd")
        memsize *= enlargement
        hidsize *= enlargement
        convfeat = 16*enlargement
        self.ob_rms = RunningMeanStd(shape=list(ob_space.shape[:2])+[1], use_mpi=not update_ob_stats_independently_per_gpu)
        ph_istate = tf.placeholder(dtype=tf.float32,shape=(None,memsize), name='state')
        pdparamsize = self.pdtype.param_shape()[0]
        self.memsize = memsize

        #Inputs to policy and value function will have different shapes depending on whether it is rollout
        #or optimization time, so we treat separately.
        self.pdparam_opt, self.vpred_int_opt, self.vpred_ext_opt, self.snext_opt = \
            self.apply_policy(self.ph_ob[None][:,:-1],
                              reuse=False,
                              scope=scope,
                              hidsize=hidsize,
                              memsize=memsize,
                              extrahid=extrahid,
                              sy_nenvs=self.sy_nenvs,
                              sy_nsteps=self.sy_nsteps - 1,
                              pdparamsize=pdparamsize
                              )
        self.pdparam_rollout, self.vpred_int_rollout, self.vpred_ext_rollout, self.snext_rollout = \
            self.apply_policy(self.ph_ob[None],
                              reuse=True,
                              scope=scope,
                              hidsize=hidsize,
                              memsize=memsize,
                              extrahid=extrahid,
                              sy_nenvs=self.sy_nenvs,
                              sy_nsteps=self.sy_nsteps,
                              pdparamsize=pdparamsize
                              )
        if dynamics_bonus:
            self.define_dynamics_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)
        else:
            self.define_self_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)

        pd = self.pdtype.pdfromflat(self.pdparam_rollout)
        self.a_samp = pd.sample()
        self.nlp_samp = pd.neglogp(self.a_samp)
        self.entropy_rollout = pd.entropy()
        self.pd_rollout = pd

        self.pd_opt = self.pdtype.pdfromflat(self.pdparam_opt)

        self.ph_istate = ph_istate

    @staticmethod
    def apply_policy(ph_ob, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize):
        data_format = 'NHWC'
        ph = ph_ob
        assert len(ph.shape.as_list()) == 5  # B,T,H,W,C
        logger.info("CnnPolicy: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        X = tf.cast(ph, tf.float32) / 255.
        X = tf.reshape(X, (-1, *ph.shape.as_list()[-3:]))

        activ = tf.nn.relu
        yes_gpu = any(get_available_gpus())
        with tf.variable_scope(scope, reuse=reuse), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):
            X = activ(conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), data_format=data_format))
            X = to2d(X)
            mix_other_observations = [X]
            X = tf.concat(mix_other_observations, axis=1)
            X = activ(fc(X, 'fc1', nh=hidsize, init_scale=np.sqrt(2)))
            additional_size = 448
            X = activ(fc(X, 'fc_additional', nh=additional_size, init_scale=np.sqrt(2)))
            snext = tf.zeros((sy_nenvs, memsize))
            mix_timeout = [X]

            Xtout = tf.concat(mix_timeout, axis=1)
            if extrahid:
                Xtout = X + activ(fc(Xtout, 'fc2val', nh=additional_size, init_scale=0.1))
                X     = X + activ(fc(X, 'fc2act', nh=additional_size, init_scale=0.1))
            pdparam = fc(X, 'pd', nh=pdparamsize, init_scale=0.01)
            vpred_int   = fc(Xtout, 'vf_int', nh=1, init_scale=0.01)
            vpred_ext   = fc(Xtout, 'vf_ext', nh=1, init_scale=0.01)

            pdparam = tf.reshape(pdparam, (sy_nenvs, sy_nsteps, pdparamsize))
            vpred_int = tf.reshape(vpred_int, (sy_nenvs, sy_nsteps))
            vpred_ext = tf.reshape(vpred_ext, (sy_nenvs, sy_nsteps))
        return pdparam, vpred_int, vpred_ext, snext

    def define_self_prediction_rew(self, convfeat, rep_size, enlargement):
        logger.info("Using RND BONUS ****************************************************")

        #RND bonus.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]
                X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:,1:]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)
                # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        targets = tf.stop_gradient(X_r)
        # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        self.aux_loss = tf.reduce_mean(tf.square(targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

    def define_dynamics_prediction_rew(self, convfeat, rep_size, enlargement):
        #Dynamics loss with random features.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]
                X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        ac_one_hot = tf.one_hot(self.ph_ac, self.ac_space.n, axis=2)
        assert ac_one_hot.get_shape().ndims == 3
        assert ac_one_hot.get_shape().as_list() == [None, None, self.ac_space.n], ac_one_hot.get_shape().as_list()
        ac_one_hot = tf.reshape(ac_one_hot, (-1, self.ac_space.n))
        def cond(x):
            return tf.concat([x, ac_one_hot], 1)

        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:,:-1]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))
                # ph_mean, ph_std are 84x84x1, so we subtract the average of the last channel from all channels. Is this ok?
                xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)

                # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(cond(rgbrp), 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(cond(X_r_hat), 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = fc(cond(X_r_hat), 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        noisy_targets = tf.stop_gradient(X_r)
        # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        self.aux_loss = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

    def initial_state(self, n):
        return np.zeros((n, self.memsize), np.float32)

    def call(self, dict_obs, new, istate, update_obs_stats=False):
        for ob in dict_obs.values():
            if ob is not None:
                if update_obs_stats:
                    raise NotImplementedError
                    ob = ob.astype(np.float32)
                    ob = ob.reshape(-1, *self.ob_space.shape)
                    self.ob_rms.update(ob)
        # Note: if it fails here with ph vs observations inconsistency, check if you're loading agent from disk.
        # It will use whatever observation spaces saved to disk along with other ctor params.
        feed1 = { self.ph_ob[k]: dict_obs[k][:,None] for k in self.ph_ob_keys }
        feed2 = { self.ph_istate: istate, self.ph_new: new[:,None].astype(np.float32) }
        feed1.update({self.ph_mean: self.ob_rms.mean, self.ph_std: self.ob_rms.var ** 0.5})
        # for f in feed1:
        #     print(f)
        a, vpred_int,vpred_ext, nlp, newstate, ent = tf.get_default_session().run(
            [self.a_samp, self.vpred_int_rollout,self.vpred_ext_rollout, self.nlp_samp, self.snext_rollout, self.entropy_rollout],
            feed_dict={**feed1, **feed2})
        return a[:,0], vpred_int[:,0],vpred_ext[:,0], nlp[:,0], newstate, ent[:,0]
