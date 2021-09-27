#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 23:46:51 2021
@author: orman
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import vi as tfv
from tensorflow_probability.python  import monte_carlo
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal.reparameterization import FULLY_REPARAMETERIZED

        
def renyi_divergence(target_log_prob_fn,
                     surrogate_posterior,
                     sample_size = 1,
                     alpha = 0.999,
                     reverse = True,
                     use_reparameterization = None,
                     seed = None,
                     name = None):
    
    with tf.name_scope(name or 'monte_carlo_renyi_divergence'):
        reparameterization_types = tf.nest.flatten(surrogate_posterior.reparameterization_type)
        if use_reparameterization is None:
            use_reparameterization = all(
                reparameterization_type == FULLY_REPARAMETERIZED for 
                reparameterization_type in reparameterization_types)
        elif (use_reparameterization and
              any(reparameterization_type != FULLY_REPARAMETERIZED
                  for reparameterization_type in reparameterization_types)):
            raise ValueError(
                'Distribution `surrogate_posterior` must be reparameterized, i.e.,'
                'a diffeomorphic transformation of a parameterless distribution. '
                '(Otherwise this function has a biased gradient.)')
        if not callable(target_log_prob_fn):
            raise TypeError('`target_log_prob_fn` must be a Python `callable` function.')
                    
        def div_fn_wrap(q_samples, fn):
            target_log_prob = nest_util.call_fn(target_log_prob_fn, q_samples)
            return fn(target_log_prob - surrogate_posterior.log_prob(q_samples))

        if alpha == 1.0:
            if reverse:
                div_fn = lambda q_samples: div_fn_wrap(q_samples, fn = tfv.kl_reverse)
            else:
                div_fn = lambda q_samples: div_fn_wrap(q_samples, fn = tfv.kl_forward)
        else:
            if reverse:
                factor = 1.0 - alpha
            else:
                factor = alpha
            renyi = lambda logu: factor * logu 
            div_fn = lambda q_samples: div_fn_wrap(q_samples, fn = renyi)
        
        q_samples = surrogate_posterior.sample(sample_size, seed = seed)
        
        if alpha == 1.0:
            return monte_carlo.expectation(
                f = div_fn,
                samples = q_samples,
                log_prob = surrogate_posterior.log_prob,
                use_reparameterization = use_reparameterization)
        else:
            return 1/(alpha - 1) * logexpectationexp(
                f = div_fn,
                samples = q_samples,
                log_prob = surrogate_posterior.log_prob,
                use_reparameterization = use_reparameterization)


def logexpectationexp(f,
                samples,
                log_prob=None,
                use_reparameterization=True,
                axis=0,
                keepdims=False,
                name=None):
    """
    Computes the Monte-Carlo approximation of `log(E_p[exp(f(X))])`.
    
    """
    with tf.name_scope(name or 'expectation'):
        if not callable(f):
            raise ValueError('`f` must be a callable function.')
        if use_reparameterization:
            return tfp.math.reduce_logmeanexp(f(samples), axis=axis, keepdims=keepdims)
        else:
            if not callable(log_prob):
                raise ValueError('`log_prob` must be a callable function.')
            stop = tf.stop_gradient  # For readability.
            x = tf.nest.map_structure(stop, samples)
            logpx = log_prob(x)
            fx = f(x)
            dice = fx * tf.exp(logpx - stop(logpx))
            return tfp.math.reduce_logmeanexp(dice, axis=axis, keepdims=keepdims)