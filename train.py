import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from renyi_divergence import renyi_divergence

import config as conf

from data.mnist import Mnist
from models.vae import VAE


#%%
def make_train_procedure(name, model, opt):
    @tf.function
    def train_step_kl(x):
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            z = model.get_post_dist(mu, log_sigma).sample()
            f_z = model.decoder(z, training=True)
            
            encoder_loss = model.encoder_loss(mu, log_sigma)
            decoder_loss = model.decoder_loss(x, f_z)
            
            loss = encoder_loss + decoder_loss
    
            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables))
    
            return encoder_loss, decoder_loss, loss
    
    @tf.function
    def train_step_kl_mc(x):
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            z = model.get_post_dist(mu, log_sigma).sample()
            f_z = model.decoder(z, training=True)
            
            encoder_loss = model.encoder_loss_mc(mu, log_sigma, conf.SAMPLE_SIZE)
            decoder_loss = model.decoder_loss(x, f_z)
            
            loss = encoder_loss + decoder_loss
    
            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables))
    
            return encoder_loss, decoder_loss, loss
    
    @tf.function
    def train_step_kl_full(x):
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            post_dist = model.get_post_dist(mu, log_sigma)
            
            container = {}
            
            def log_prob(q_samples, x, container):
                log_prior = tfd.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(mu)).log_prob(q_samples)
                f_z = model.decode(q_samples, training = True)
                log_like = tfd.Independent(tfd.Normal(f_z, 1), 1).log_prob(x)
                container['decoder_loss'] = -tf.reduce_mean(log_like)
                return log_prior + log_like
            
            target_log_prob_fn = lambda q_samples: log_prob(q_samples, x, container)
            
            vfe = tfp.vi.monte_carlo_variational_loss(
                target_log_prob_fn = target_log_prob_fn,
                surrogate_posterior = post_dist,
                sample_size = conf.SAMPLE_SIZE,
                discrepancy_fn = tfp.vi.kl_reverse)

            loss = tf.reduce_mean(vfe)
            
            encoder_loss = model.encoder_loss_mc(mu, log_sigma, conf.SAMPLE_SIZE)
            decoder_loss = container['decoder_loss']
            
            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables))
    
            return encoder_loss, decoder_loss, loss   
    
    
    @tf.function
    def train_step_renyi(x):
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            post_dist = model.get_post_dist(mu, log_sigma)
            
            container = {}
            
            def log_prob(q_samples, x, container):
                log_prior = tfd.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(mu)).log_prob(q_samples)
                f_z = model.decode(q_samples, training = True)
                log_like = tfd.Independent(tfd.Normal(f_z, 1), 1).log_prob(x)
                container['decoder_loss'] = -tf.reduce_mean(log_like)
                return log_prior + log_like
            
            target_log_prob_fn = lambda q_samples: log_prob(q_samples, x, container)
            
            vfe = renyi_divergence(
                target_log_prob_fn = target_log_prob_fn,
                surrogate_posterior = post_dist,
                sample_size = conf.SAMPLE_SIZE,
                alpha = conf.ALPHA)
            
            loss = tf.reduce_mean(vfe)
            
            encoder_loss = model.encoder_loss_mc(mu, log_sigma, conf.SAMPLE_SIZE)
            decoder_loss = container['decoder_loss']
            
            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables))
    
            return encoder_loss, decoder_loss, loss

    @tf.function
    def train_step_hellinger(x):
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            post_dist = model.get_post_dist(mu, log_sigma)
            
            container = {}
            
            def log_prob(q_samples, x, container):
                log_prior = tfd.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(mu)).log_prob(q_samples)
                f_z = model.decode(q_samples, training = True)
                log_like = tfd.Independent(tfd.Normal(f_z, 1), 1).log_prob(x)
                container['decoder_loss'] = -tf.reduce_mean(log_like)
                return log_prior + log_like
            
            target_log_prob_fn = lambda q_samples: log_prob(q_samples, x, container)
            
            vfe = tfp.vi.monte_carlo_variational_loss(
                target_log_prob_fn = target_log_prob_fn,
                surrogate_posterior = post_dist,
                sample_size = conf.SAMPLE_SIZE,
                discrepancy_fn = tfp.vi.squared_hellinger)
            
            loss = tf.reduce_mean(vfe)
            
            encoder_loss = model.encoder_loss_mc(mu, log_sigma, conf.SAMPLE_SIZE)
            decoder_loss = container['decoder_loss']
            
            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables))
    
            return encoder_loss, decoder_loss, loss
    
    if name.lower() in ('default', 'kl'):
        fn = train_step_kl
        label = 'kl'
    elif name.lower() in ('kl_mc',):
        fn = train_step_kl_mc
        label = 'kl_mc_{0}'.format(conf.SAMPLE_SIZE)
    elif name.lower() in ('kl_full',):
        fn = train_step_kl_full
        label = 'kl_full_{0}'.format(conf.SAMPLE_SIZE)
    elif name.lower() in ('renyi',):
        fn =  train_step_renyi
        label = 'renyi_{0}_{1}'.format(conf.SAMPLE_SIZE, conf.ALPHA)
    elif name.lower() in ('hellinger',):
        fn = train_step_hellinger
        label = 'hellinger_{0}'.format(conf.SAMPLE_SIZE)

    return fn, label


#%%
class Driver():
    def __init__(self):
        loader = Mnist()
        self.features = np.vstack([loader.train_features, loader.test_features]).astype(np.float32)    
        self.num_sets = loader.num_train_sets + loader.num_test_sets
        
        feature_depth = loader.feature_depth    
        self.model = VAE(conf.LATENT_DEPTH, feature_depth)
    
        self.opt = tf.keras.optimizers.Adam()
        
        self.ckpt = tf.train.Checkpoint(encoder=self.model.encoder, decoder=self.model.decoder)

        self.step_func, self.label = make_train_procedure(conf.TRAIN_STEP_FUNCTION, self.model, self.opt)
        self.model_name = '{0}_{1}'.format(conf.MODEL_NAME, self.label)
        
        model_ckpt_name = '{0}_model_ckpt'.format(self.model_name)
        model_spec_name = '{0}_model_spec.json'.format(self.model_name)
        model_rslt_name = '{0}_results.pickle'.format(self.model_name)

        model_save_path = os.path.join(conf.MODEL_SAVE_DIR, self.model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        self.model_ckpt_path = os.path.join(model_save_path, model_ckpt_name)
        self.model_spec_path = os.path.join(model_save_path, model_spec_name)
        self.model_rslt_path = os.path.join(model_save_path, model_rslt_name)

    def __call__(self, num_epochs: int):
        steps_per_epoch = self.num_sets // conf.BATCH_SIZE
        train_steps = steps_per_epoch * num_epochs
        
        print('steps_per_epoch: {0}, train_steps: {1}'.format(steps_per_epoch, train_steps))
        
        encoder_losses = []
        decoder_losses = []
        losses = []
        encoder_losses_epoch = []
        decoder_losses_epoch = []
        losses_epoch = []
        fs = []
        for i in range(1, train_steps+1):
            epoch = i // steps_per_epoch
    
            idxes = np.random.choice(self.num_sets, conf.BATCH_SIZE, replace=False)
            x_i = self.features[idxes]
            eps_i = np.random.normal(size=[conf.BATCH_SIZE, conf.LATENT_DEPTH]).astype(np.float32)
            
            encoder_loss_i, decoder_loss_i, loss_i = self.step_func(x_i)
            
            encoder_losses.append(encoder_loss_i)
            decoder_losses.append(decoder_loss_i)
            losses.append(loss_i)
    
            if i % steps_per_epoch == 0:
                f_eps = self.model.decode(eps_i, training=False)
    
                encoder_loss_epoch = np.mean(encoder_losses[-steps_per_epoch:])
                decoder_loss_epoch = np.mean(decoder_losses[-steps_per_epoch:])
                loss_epoch = np.mean(losses[-steps_per_epoch:])
    
                print("Epoch: %i,  Encoder Loss: %f,  Decoder Loss: %f, Loss: %f" % \
                    (epoch, encoder_loss_epoch, decoder_loss_epoch, loss_epoch)
                )
    
                encoder_losses_epoch.append(encoder_loss_epoch)
                decoder_losses_epoch.append(decoder_loss_epoch)
                losses_epoch.append(loss_epoch)
    
                fs.append(f_eps)
                
                self.ckpt.save(file_prefix=self.model_ckpt_path)
    
                with open(self.model_rslt_path, "wb") as f:
                    pickle.dump((encoder_losses_epoch, decoder_losses_epoch, losses_epoch, fs), f)


#%%
if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(False)
    driver = Driver()
    driver(100)