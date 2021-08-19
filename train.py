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


def main():
    model_spec_name = "%s-model-spec.json" % conf.MODEL_NAME
    model_rslt_name = "%s-results.pickle" % conf.MODEL_NAME

    model_save_path = os.path.join(conf.MODEL_SAVE_DIR, conf.MODEL_NAME)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_ckpt_path = os.path.join(model_save_path, "model-ckpt")
    model_spec_path = os.path.join(model_save_path, model_spec_name)
    model_rslt_path = os.path.join(model_save_path, model_rslt_name)

    loader = Mnist()

    features = np.vstack([loader.train_features, loader.test_features]).astype(np.float32)

    num_sets = loader.num_train_sets + loader.num_test_sets
    
    feature_depth = loader.feature_depth
    feature_shape = loader.feature_shape

    latent_depth = conf.LATENT_DEPTH

    batch_size = conf.BATCH_SIZE
    num_epochs = conf.NUM_EPOCHS

    model = VAE(latent_depth, feature_depth)

    opt = tf.keras.optimizers.Adam(learning_rate = 2)

#%%
    @tf.function
    def train_step(x, eps):
        """
        Original trianing step code

        """
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            z = model.reparam(eps, mu, log_sigma)
            f_z = model.decode(z, training=True)

            encoder_loss = tf.reduce_mean(model.encoder_loss(mu, log_sigma))
            decoder_loss = tf.reduce_mean(model.decoder_loss(x, f_z))
            loss = encoder_loss + decoder_loss

            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables)
            )

        return encoder_loss, decoder_loss, loss

#%%
    @tf.function
    def train_step_renyi(x, eps):
        """
        Renyi loss

        """
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            post_dist = tfd.MultivariateNormalDiag(mu, tf.exp(log_sigma))
            
            container = {
                'encoder_loss': tf.reduce_mean(model.encoder_loss(mu, log_sigma))
                }
            
            def log_prob(q_samples, x, container):
                log_prior = tfd.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(mu)).log_prob(q_samples)
                mean = model.decode(q_samples, training=True)
                log_like = tfd.Independent(tfd.Normal(mean, 1), 1).log_prob(x)
                container['decoder_loss'] = tf.reduce_mean(model.decoder_loss(x, mean))
                return log_prior + log_like
            
            target_log_prob_fn = lambda q_samples: log_prob(q_samples, x, container)
            
            loss = renyi_divergence(
                target_log_prob_fn = target_log_prob_fn,
                surrogate_posterior = post_dist,
                sample_size = 100,
                alpha = 1.0, #change alpha here
                )

            loss = tf.reduce_mean(loss)
            
            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables)
            )

        return container['encoder_loss'], container['decoder_loss'], loss
    
#%%
    @tf.function
    def train_step_tdiv(x, eps):
        """
        T divergence loss

        """
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            post_dist = tfd.MultivariateNormalDiag(mu, tf.exp(log_sigma))
            
            container = {
                'encoder_loss': tf.reduce_mean(model.encoder_loss(mu, log_sigma))
                }
            
            def log_prob(q_samples, x, container):
                log_prior = tfd.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(mu)).log_prob(q_samples)
                mean = model.decode(q_samples, training=True)
                log_like = tfd.Independent(tfd.Normal(mean, 1), 1).log_prob(x)
                container['decoder_loss'] = tf.reduce_mean(model.decoder_loss(x, mean))
                return log_prior + log_like
            
            target_log_prob_fn = lambda q_samples: log_prob(q_samples, x, container)
            discrepancy_fn = lambda logu:  tfp.vi.amari_alpha(logu, alpha = 0.5) #change alpha here
            
            loss = tfp.vi.monte_carlo_variational_loss(
                target_log_prob_fn = target_log_prob_fn,
                surrogate_posterior = post_dist,
                sample_size = 100,
                discrepancy_fn = discrepancy_fn)  

            loss = tf.reduce_mean(loss)
            
            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables)
            )

        return container['encoder_loss'], container['decoder_loss'], loss

#%%
    @tf.function
    def train_step_KL(x, eps):
        """
        KL with likelihood expanded out of the KL.
        
        """
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            post_dist = tfd.MultivariateNormalDiag(mu, tf.exp(log_sigma))
            
            def log_prob(q_samples, x):
                log_prior = tfd.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(mu)).log_prob(q_samples)
                return log_prior
            
            target_log_prob_fn = lambda q_samples: log_prob(q_samples, x)
            
            kl_val = tfp.vi.monte_carlo_variational_loss(
                target_log_prob_fn = target_log_prob_fn,
                surrogate_posterior = post_dist,
                sample_size = 100,
                discrepancy_fn = tfp.vi.kl_reverse)
            
            kl_loss = tf.reduce_mean(kl_val)
            
            q_sample = post_dist.sample()
            mean = model.decode(q_sample, training=True)
            log_like = tfd.Independent(tfd.Normal(mean, 1), 1).log_prob(x)
            
            image_loss = tf.reduce_mean(log_like)
            
            loss = kl_loss - image_loss
            
            encoder_loss = tf.reduce_mean(model.encoder_loss(mu, log_sigma))
            decoder_loss = tf.reduce_mean(model.decoder_loss(x, mean))
            
            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables)
            )

        return encoder_loss, decoder_loss, loss

#%%
    @tf.function
    def train_step_KL_full(x, eps):
        """
        KL with likelihood included in the KL
        
        """
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            post_dist = tfd.MultivariateNormalDiag(mu, tf.exp(log_sigma))
            
            container = {
                'encoder_loss': tf.reduce_mean(model.encoder_loss(mu, log_sigma))
                }
            
            def log_prob(q_samples, x, container):
                log_prior = tfd.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(mu)).log_prob(q_samples)
                mean = model.decode(q_samples, training=True)
                log_like = tfd.Independent(tfd.Normal(mean, 1), 1).log_prob(x)
                container['decoder_loss'] = tf.reduce_mean(model.decoder_loss(x, mean))
                return log_prior + log_like
            
            target_log_prob_fn = lambda q_samples: log_prob(q_samples, x, container)

            loss = tfp.vi.monte_carlo_variational_loss(
                target_log_prob_fn = target_log_prob_fn,
                surrogate_posterior = post_dist,
                sample_size = 100,
                discrepancy_fn = tfp.vi.kl_reverse)

            loss = tf.reduce_mean(loss)
            
            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables)
            )

        return container['encoder_loss'], container['decoder_loss'], loss

#%%
    ckpt = tf.train.Checkpoint(encoder=model.encoder, decoder=model.decoder)

    steps_per_epoch = num_sets // batch_size
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

        idxes = np.random.choice(num_sets, batch_size, replace=False)
        x_i = features[idxes]
        eps_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)
        
        #%%Change here to use different functions
        encoder_loss_i, decoder_loss_i, loss_i = train_step_renyi(x_i, eps_i)
        
        encoder_losses.append(encoder_loss_i)
        decoder_losses.append(decoder_loss_i)
        losses.append(loss_i)

        if i % steps_per_epoch == 0:
            f_eps = model.decode(eps_i, training=False)

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
            
            ckpt.save(file_prefix=model_ckpt_path)

            with open(model_rslt_path, "wb") as f:
                pickle.dump((encoder_losses_epoch, decoder_losses_epoch, losses_epoch, fs), f)


#%%
if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(False)
    tf.random.set_seed(34057230946709324760923740672)

    main()
